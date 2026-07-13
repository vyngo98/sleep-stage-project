from fastapi import FastAPI
import tensorflow as tf
from tfserving_client import *
from data_processing import *
from scipy.ndimage import median_filter

app = FastAPI()

@app.post("/sleep-stage/predict")
def predict_sleep_stage(req: SleepStageRequest):
    request = predict_pb2.PredictRequest()

    request.model_spec.name = (
        "sleepstage-serving"
    )

    request.model_spec.signature_name = (
        "serving_default"
    )

    (ihr_segments, clean_activity_count_long_segments, total_label_segs,
     start_time_sleep, seg_indices, ihr_signals_org) = generator_test_single_hsat_padding(DATA_FOLDER,
                                                                                          req.record_id, N_CLASSES,
                                                                                          n_epochs=250,
                                                                                          stride_size=int(30 * 60))
    y_predict = []
    for batch in range(0, len(ihr_segments), MAX_BATCHSIZE):
        batch_ihr_segments = ihr_segments[batch:batch + MAX_BATCHSIZE]
        batch_activity_count_segments = clean_activity_count_long_segments[batch:batch + MAX_BATCHSIZE]

        # Convert input to TensorProto
        proto = tf.make_tensor_proto(
            batch_ihr_segments, dtype=tf.float32, shape=batch_ihr_segments.shape)
        request.inputs['keras_tensor'].CopyFrom(proto)

        proto2 = tf.make_tensor_proto(
            batch_activity_count_segments, dtype=tf.float32, shape=batch_activity_count_segments.shape)
        request.inputs['keras_tensor_1'].CopyFrom(proto2)

        ## Retry loop
        while True:
            try:
                response = stub.Predict(request, timeout=10)
                output = np.array(response.outputs['output_0'].float_val)
                predict = np.reshape(output, (np.shape(batch_ihr_segments)[0], np.shape(batch_ihr_segments)[1], N_CLASSES))
                y_predict.extend(predict)
                break
            except Exception as e:
                print(
                    f"[sleepstage-serving] Waiting for TF server... Retrying in 1s. Error: {e}")

    y_predict = np.array(y_predict)

    ihr_segments_squeeze = np.squeeze(ihr_segments)
    cont_ihr_segs = []
    for n in range(len(ihr_segments)):
        ihr_seg = ihr_segments_squeeze[n]
        cont_ihr_seg = get_continuous_signal_stride(ihr_seg, stride=30, fs=FS_HR)
        cont_ihr_segs.append(cont_ihr_seg)

    cont_ihr_segs = np.array(cont_ihr_segs)
    cont_ihr = reconstruct_continuous_signal(cont_ihr_segs, seg_indices, stride=int(30 * 60), fs=FS_HR)

    cont_ihr_org = reconstruct_continuous_signal(ihr_signals_org, seg_indices, stride=int(30 * 60), fs=FS_HR)

    final_prediction = merge_predictions_weighted_reconstruct(y_predict, seg_indices, stride_sec=int(30 * 60),
                                                              seg_len_sec=7500, n_classes=3, epoch_sec=30)

    cont_label_segs = reconstruct_continuous_signal(total_label_segs, seg_indices, stride=int(30 * 60), fs=1 / 30)

    new_filtered_pred = post_process_sleep_labels(final_prediction, epoch_sec=30,

                                                  # thresholds
                                                  min_wake_sec=30,
                                                  min_rem_sec=90,

                                                  # hole filling
                                                  max_gap_sec=330,

                                                  # wake inside rem
                                                  wake_inside_rem_sec=120, )

    # find long 0 ihr duration in the end
    pad_len_ihr = trailing_neg_length(cont_ihr_org, value=0)
    pad_len = trailing_neg_length(cont_label_segs, value=-1)
    pad_len = max(pad_len, int(pad_len_ihr / 2 / 30))
    orginal_len = len(cont_label_segs) - pad_len

    # Remove padding prediction
    cont_label_segs = cont_label_segs[:orginal_len]
    new_filtered_pred = new_filtered_pred[: orginal_len]

    # Remove when label has long nan duration 2026-04-05-PhuLe-4mm-10mA
    nonnan_ind = find_before_long_nan(cont_label_segs, min_nan_len=150)
    if len(nonnan_ind) > 0:
        cont_label_segs = cont_label_segs[:nonnan_ind[0] + 1]
        new_filtered_pred = new_filtered_pred[:nonnan_ind[0] + 1]

    cont_label_segs_ = np.repeat(cont_label_segs, 30)
    final_prediction_smooth_ = np.repeat(new_filtered_pred, 30)

    # mask valid positions
    valid_mask = (~np.isnan(cont_label_segs_))

    # remove NaN positions
    y_true = cont_label_segs_[valid_mask]
    final_prediction_smooth = final_prediction_smooth_[valid_mask]

    # optional: convert to int
    cont_label_segs = y_true.astype(int)
    final_prediction_smooth = final_prediction_smooth.astype(int)

    [idx, percent_rem_true, percent_rem_pred, percent_nrem_true, percent_nrem_pred,
     percent_wake_true, percent_wake_pred, ratio_true, ratio_pred, episode_true, episode_pred, latency_true,
     latency_pred] = architecture_metrics(req.record_id, cont_label_segs, final_prediction_smooth)

    return {
        "record_id": req.record_id,

        "sleep_stage":
            new_filtered_pred.tolist(),

        "sleep_stage_smooth":
            final_prediction_smooth.tolist(),

        "metrics": {

            "percent_rem_pred":
                float(percent_rem_pred),

            "percent_nrem_pred":
                float(percent_nrem_pred),

            "percent_wake_pred":
                float(percent_wake_pred),

            "latency_pred":
                latency_pred
        }
    }

