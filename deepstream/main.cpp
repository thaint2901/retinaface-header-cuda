#include <gst/gst.h>
#include <glib.h>
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>
/* Open CV headers */
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "nvbufsurface.h"
#include "cuda_runtime_api.h"
#include "nvdsinfer_custom_impl.h"
#include "gstnvdsmeta.h"
#include "nvdsmeta_schema.h"
#include "gstnvdsinfer.h"
#include "nvds_version.h"

#include "aligner.h"
#include "base64.h"
#include "protobuf.pb.h"

#define INFER_PGIE_CONFIG_FILE  "/nvidia/retinaface-header-cuda/deepstream/configs/infer_config_batch1.txt"
#define INFER_SGIE1_CONFIG_FILE "/nvidia/retinaface-header-cuda/deepstream/configs/dstensor_sgie_config.txt"
#define MSCONV_CONFIG_FILE "/nvidia/retinaface-header-cuda/deepstream/configs/dstest4_msgconv_config.txt"
#define TRACKER_CONFIG_FILE "/nvidia/retinaface-header-cuda/deepstream/configs/dstest2_tracker_config.txt"

#define MAX_TIME_STAMP_LEN 32
#define MUXER_OUTPUT_WIDTH 1920
#define MUXER_OUTPUT_HEIGHT 1080

#define PGIE_NET_WIDTH 960
#define PGIE_NET_HEIGHT 540

/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 40000

/* NVIDIA Decoder source pad memory feature. This feature signifies that source
 * pads having this capability will push GstBuffers containing NvBufSurface. */
#define MEMORY_FEATURES "memory:NVMM"

unsigned int nvds_lib_major_version = NVDS_VERSION_MAJOR;
unsigned int nvds_lib_minor_version = NVDS_VERSION_MINOR;

gint frame_number = 0;
static gint schema_type = 0;
static gchar *cfg_file = "/nvidia/retinaface-header-cuda/deepstream/configs/cfg_kafka.txt";
static gchar *topic = "message-log2";
// static gchar *conn_str = "172.17.0.1;9092";
static gchar *conn_str = "103.226.250.14;9092";
static gchar *proto_lib = "/opt/nvidia/deepstream/deepstream-5.0/lib/libnvds_kafka_proto.so";
static gchar *msg2p_lib = "/nvidia/retinaface-header-cuda/deepstream/nvmsgconv/libnvds_msgconv.so";

extern "C"
  bool NvDsInferParseRetinaNet (std::vector < NvDsInferLayerInfo >
  const &outputLayersInfo, NvDsInferNetworkInfo const &networkInfo,
  NvDsInferParseDetectionParams const &detectionParams,
  std::vector < NvDsInferObjectDetectionInfo > &objectList);

#define CONFIG_GROUP_TRACKER "tracker"
#define CONFIG_GROUP_TRACKER_WIDTH "tracker-width"
#define CONFIG_GROUP_TRACKER_HEIGHT "tracker-height"
#define CONFIG_GROUP_TRACKER_LL_CONFIG_FILE "ll-config-file"
#define CONFIG_GROUP_TRACKER_LL_LIB_FILE "ll-lib-file"
#define CONFIG_GROUP_TRACKER_ENABLE_BATCH_PROCESS "enable-batch-process"
#define CONFIG_GPU_ID "gpu-id"

static gchar *
get_absolute_file_path (gchar *cfg_file_path, gchar *file_path)
{
  gchar abs_cfg_path[PATH_MAX + 1];
  gchar *abs_file_path;
  gchar *delim;

  if (file_path && file_path[0] == '/') {
    return file_path;
  }

  if (!realpath (cfg_file_path, abs_cfg_path)) {
    g_free (file_path);
    return NULL;
  }

  // Return absolute path of config file if file_path is NULL.
  if (!file_path) {
    abs_file_path = g_strdup (abs_cfg_path);
    return abs_file_path;
  }

  delim = g_strrstr (abs_cfg_path, "/");
  *(delim + 1) = '\0';

  abs_file_path = g_strconcat (abs_cfg_path, file_path, NULL);
  g_free (file_path);

  return abs_file_path;
}

/* Tracker config parsing */

#define CHECK_ERROR(error) \
    if (error) { \
        g_printerr ("Error while parsing config file: %s\n", error->message); \
        goto done; \
    }

static gboolean
set_tracker_properties (GstElement *nvtracker)
{
  gboolean ret = FALSE;
  GError *error = NULL;
  gchar **keys = NULL;
  gchar **key = NULL;
  GKeyFile *key_file = g_key_file_new ();

  if (!g_key_file_load_from_file (key_file, TRACKER_CONFIG_FILE, G_KEY_FILE_NONE,
          &error)) {
    g_printerr ("Failed to load config file: %s\n", error->message);
    return FALSE;
  }

  keys = g_key_file_get_keys (key_file, CONFIG_GROUP_TRACKER, NULL, &error);
  CHECK_ERROR (error);

  for (key = keys; *key; key++) {
    if (!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_WIDTH)) {
      gint width =
          g_key_file_get_integer (key_file, CONFIG_GROUP_TRACKER,
          CONFIG_GROUP_TRACKER_WIDTH, &error);
      CHECK_ERROR (error);
      g_object_set (G_OBJECT (nvtracker), "tracker-width", width, NULL);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_HEIGHT)) {
      gint height =
          g_key_file_get_integer (key_file, CONFIG_GROUP_TRACKER,
          CONFIG_GROUP_TRACKER_HEIGHT, &error);
      CHECK_ERROR (error);
      g_object_set (G_OBJECT (nvtracker), "tracker-height", height, NULL);
    } else if (!g_strcmp0 (*key, CONFIG_GPU_ID)) {
      guint gpu_id =
          g_key_file_get_integer (key_file, CONFIG_GROUP_TRACKER,
          CONFIG_GPU_ID, &error);
      CHECK_ERROR (error);
      g_object_set (G_OBJECT (nvtracker), "gpu_id", gpu_id, NULL);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_LL_CONFIG_FILE)) {
      char* ll_config_file = get_absolute_file_path (TRACKER_CONFIG_FILE,
                g_key_file_get_string (key_file,
                    CONFIG_GROUP_TRACKER,
                    CONFIG_GROUP_TRACKER_LL_CONFIG_FILE, &error));
      CHECK_ERROR (error);
      g_object_set (G_OBJECT (nvtracker), "ll-config-file", ll_config_file, NULL);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_LL_LIB_FILE)) {
      char* ll_lib_file = get_absolute_file_path (TRACKER_CONFIG_FILE,
                g_key_file_get_string (key_file,
                    CONFIG_GROUP_TRACKER,
                    CONFIG_GROUP_TRACKER_LL_LIB_FILE, &error));
      CHECK_ERROR (error);
      g_object_set (G_OBJECT (nvtracker), "ll-lib-file", ll_lib_file, NULL);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_ENABLE_BATCH_PROCESS)) {
      gboolean enable_batch_process =
          g_key_file_get_integer (key_file, CONFIG_GROUP_TRACKER,
          CONFIG_GROUP_TRACKER_ENABLE_BATCH_PROCESS, &error);
      CHECK_ERROR (error);
      g_object_set (G_OBJECT (nvtracker), "enable_batch_process",
                    enable_batch_process, NULL);
    } else {
      g_printerr ("Unknown key '%s' for group [%s]", *key,
          CONFIG_GROUP_TRACKER);
    }
  }

  ret = TRUE;
done:
  if (error) {
    g_error_free (error);
  }
  if (keys) {
    g_strfreev (keys);
  }
  if (!ret) {
    g_printerr ("%s failed", __func__);
  }
  return ret;
}

static void generate_ts_rfc3339 (char *buf, int buf_size)
{
  time_t tloc;
  struct tm tm_log;
  struct timespec ts;
  char strmsec[6]; //.nnnZ\0

  clock_gettime(CLOCK_REALTIME,  &ts);
  memcpy(&tloc, (void *)(&ts.tv_sec), sizeof(time_t));
  gmtime_r(&tloc, &tm_log);
  strftime(buf, buf_size,"%Y-%m-%dT%H:%M:%S", &tm_log);
  int ms = ts.tv_nsec/1000000;
  g_snprintf(strmsec, sizeof(strmsec),".%.3dZ", ms);
  strncat(buf, strmsec, buf_size);
}

static GstPadProbeReturn pgie_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info, gpointer u_data) {
  NvBufSurface *surface = NULL;
  GstBuffer * inbuf = GST_PAD_PROBE_INFO_BUFFER(info);
  GstMapInfo in_map_info;
  NvDsMetaList * l_obj = NULL;
  NvDsObjectMeta *obj_meta = NULL;
  memset (&in_map_info, 0, sizeof (in_map_info));
  if (!gst_buffer_map (inbuf, &in_map_info, GST_MAP_READ)) {
    g_error ("Error: Failed to map gst buffer\n");
  }
  surface = (NvBufSurface *) in_map_info.data;
  mirror::Aligner aligner;

  static guint use_device_mem = 0;
  static NvDsInferNetworkInfo networkInfo {PGIE_NET_WIDTH, PGIE_NET_HEIGHT, 3};
  NvDsInferParseDetectionParams detectionParams;
  detectionParams.perClassThreshold = {0.95};
  static float groupThreshold = 1;
  static float groupEps = 0.2;

  NvDsBatchMeta *batch_meta = 
    gst_buffer_get_nvds_batch_meta (inbuf);
  
  if (surface->memType != NVBUF_MEM_CUDA_UNIFIED){
    g_error ("need NVBUF_MEM_CUDA_UNIFIED memory for opencv\n");
  }
  
  /* Iterate each frame metadata in batch */
  for (NvDsMetaList * l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) l_frame->data;
    cv::Mat in_mat;

    if (surface->surfaceList[frame_meta->batch_id].mappedAddr.addr[0] == NULL){
      if (NvBufSurfaceMap (surface, frame_meta->batch_id, 0, NVBUF_MAP_READ_WRITE) != 0){
        g_error ("buffer map to be accessed by CPU failed\n");
      }
    }

    /* Cache the mapped data for CPU access */
    NvBufSurfaceSyncForCpu (surface, frame_meta->batch_id, 0);

    in_mat =
      cv::Mat (surface->surfaceList[frame_meta->batch_id].planeParams.height[0],
      surface->surfaceList[frame_meta->batch_id].planeParams.width[0], CV_8UC4,
      surface->surfaceList[frame_meta->batch_id].mappedAddr.addr[0],
      surface->surfaceList[frame_meta->batch_id].planeParams.pitch[0]);

    /* Iterate user metadata in frames to search PGIE's tensor metadata */
    for (NvDsMetaList * l_user = frame_meta->frame_user_meta_list; l_user != NULL; l_user = l_user->next) {
      NvDsUserMeta *user_meta = (NvDsUserMeta *) l_user->data;
      if (user_meta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META)
        continue;
      
      /* convert to tensor metadata */
      NvDsInferTensorMeta *meta = (NvDsInferTensorMeta *) user_meta->user_meta_data;
      for (unsigned int i = 0; i < meta->num_output_layers; i++) {
        NvDsInferLayerInfo *info = &meta->output_layers_info[i];
        info->buffer = meta->out_buf_ptrs_host[i];
        if (use_device_mem && meta->out_buf_ptrs_dev[i]) {
          cudaMemcpy (meta->out_buf_ptrs_host[i], meta->out_buf_ptrs_dev[i],
            info->inferDims.numElements * 4, cudaMemcpyDeviceToHost);
        }
      }
      /* Parse output tensor and fill detection results into objectList. */
      std::vector < NvDsInferLayerInfo >outputLayersInfo (meta->output_layers_info,
        meta->output_layers_info + meta->num_output_layers);
      std::vector < NvDsInferObjectDetectionInfo > objectList;
#if NVDS_VERSION_MAJOR >= 5
      if (nvds_lib_major_version >= 5) {
        if (meta->network_info.width != networkInfo.width ||
            meta->network_info.height != networkInfo.height ||
            meta->network_info.channels != networkInfo.channels) {
          g_error ("failed to check pgie network info\n");
        }
      }
#endif
      NvDsInferParseRetinaNet(outputLayersInfo, networkInfo,
          detectionParams, objectList);
      
      int startX = 1;
      int startY = 1;
      int size = 112;
      for (auto & obj:objectList) {
        NvDsObjectMeta *obj_meta = nvds_acquire_obj_meta_from_pool (batch_meta);
        obj_meta->unique_component_id = meta->unique_id;
        obj_meta->confidence = 0.0;
        /* This is an untracked object. Set tracking_id to -1. */
        obj_meta->object_id = UNTRACKED_OBJECT_ID;
        obj_meta->class_id = 0;

        // SgieRectParams & sgie_rect_params = obj_meta->sgie_rect_params;
        NvOSD_RectParams & rect_params = obj_meta->rect_params;
        NvOSD_RectParams & sgie_rect_params = obj_meta->sgie_rect_params;
        NvOSD_TextParams & text_params = obj_meta->text_params;

        /* Assign bounding box coordinates. */
        rect_params.left = obj.left;
        rect_params.top = obj.top;
        rect_params.width = obj.width;
        rect_params.height = obj.height;
        // if ((rect_params.top + rect_params.height) >= MUXER_OUTPUT_HEIGHT) continue;

        sgie_rect_params.left = startX;
        sgie_rect_params.top = startY;
        sgie_rect_params.width = size;
        sgie_rect_params.height = size;
        // std::cout << sgie_rect_params.left << "===" << sgie_rect_params.top << std::endl;
        // std::cout << rect_params.left << "===" << rect_params.top << std::endl;

        std::vector<cv::Point2f> landmarks;
        cv::Rect face_rect = cv::Rect (startX, startY, size, size);
        startX += size;
        if ((startX + size) > MUXER_OUTPUT_WIDTH) {
          startX = 1;
          startY += size;
        }
        cv::Mat faceAligned;

        // Draw landmarks
        for (int i=0; i<5; i++) {
          cv::Point2f p1 = cv::Point(obj.landmarks[i*2], obj.landmarks[i*2+1]);
          landmarks.emplace_back(p1);
          // cv::circle(in_mat, cv::Point(obj.landmarks[i*2], obj.landmarks[i*2+1]), 2, cv::Scalar(255, 0, 0), 2);
        }

        aligner.AlignFace(in_mat, landmarks, &faceAligned);
        faceAligned.copyTo(in_mat(face_rect));
        NvBufSurfaceUnMap (surface, frame_meta->batch_id, 0);

        /* Border of width 3. */
        rect_params.border_width = 3;
        rect_params.has_bg_color = 0;
        rect_params.border_color = (NvOSD_ColorParams) {1, 0, 0, 1};

        /* display_text requires heap allocated memory. */
        text_params.display_text = g_strdup("face");
        /* Display text above the left top corner of the object. */
        text_params.x_offset = rect_params.left;
        text_params.y_offset = rect_params.top - 10;
        /* Set black background for the text. */
        text_params.set_bg_clr = 1;
        text_params.text_bg_clr = (NvOSD_ColorParams) {0, 0, 0, 1};
        /* Font face, size and color. */
        text_params.font_params.font_name = (gchar *) "Serif";
        text_params.font_params.font_size = 11;
        text_params.font_params.font_color = (NvOSD_ColorParams) {1, 1, 1, 1};
        nvds_add_obj_meta_to_frame (frame_meta, obj_meta, NULL);
      }
    }

    // NvBufSurfaceSyncForDevice (surface, frame_meta->batch_id, 0);
  }
  use_device_mem = 1 - use_device_mem;
  gst_buffer_unmap (inbuf, &in_map_info);

  return GST_PAD_PROBE_OK;
}


static gpointer meta_copy_func (gpointer data, gpointer user_data) {
  NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
  NvDsEventMsgMeta *srcMeta = (NvDsEventMsgMeta *) user_meta->user_meta_data;
  NvDsEventMsgMeta *dstMeta = NULL;

  dstMeta = (NvDsEventMsgMeta *)g_memdup (srcMeta, sizeof(NvDsEventMsgMeta));

  if (srcMeta->ts)
    dstMeta->ts = g_strdup (srcMeta->ts);

  if (srcMeta->camID)
    dstMeta->camID = g_strdup (srcMeta->camID);
  
  if (srcMeta->extMsgSize > 0) {
    NvDsFaceObject *srcObj = (NvDsFaceObject *) srcMeta->extMsg;
    NvDsFaceObject *obj = (NvDsFaceObject *) g_malloc0 (sizeof (NvDsFaceObject));
    if (srcObj->emb)
      obj->emb = g_strdup (srcObj->emb);
    
    dstMeta->extMsg = obj;
    dstMeta->extMsgSize = sizeof (NvDsFaceObject);
  }
  return dstMeta;
}


static void meta_free_func (gpointer data, gpointer user_data) {
  NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
  NvDsEventMsgMeta *srcMeta = (NvDsEventMsgMeta *) user_meta->user_meta_data;

  g_free (srcMeta->ts);

  if(srcMeta->camID) {
    g_free (srcMeta->camID);
  }

  if (srcMeta->extMsgSize > 0) {
    NvDsFaceObject *obj = (NvDsFaceObject *) srcMeta->extMsg;
    if (obj->emb)
      g_free (obj->emb);
  }
  g_free (user_meta->user_meta_data);
  user_meta->user_meta_data = NULL;
}


static GstPadProbeReturn sgie_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info, gpointer u_data) {
  GstBuffer * inbuf = GST_PAD_PROBE_INFO_BUFFER(info);

  static guint use_device_mem = 0;

  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (inbuf);

  for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame=l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) l_frame->data;

    for (NvDsMetaList * l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj=l_obj->next) {
      NvDsObjectMeta *obj_meta = (NvDsObjectMeta *) l_obj->data;

      NvOSD_RectParams & sgie_rect_params = obj_meta->sgie_rect_params;
      // std::cout << obj_meta->object_id << "===" << sgie_rect_params.left << std::endl;

      for (NvDsMetaList * l_user = obj_meta->obj_user_meta_list; l_user != NULL; l_user = l_user->next) {
        NvDsUserMeta *user_meta = (NvDsUserMeta *) l_user->data;
        if (user_meta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META)
        continue;

        /* convert to tensor metadata */
        NvDsInferTensorMeta *meta = (NvDsInferTensorMeta *) user_meta->user_meta_data;
        NvDsInferLayerInfo *info = &meta->output_layers_info[0];
        info->buffer = meta->out_buf_ptrs_host[0];
        if (use_device_mem && meta->out_buf_ptrs_dev[0]) {
          cudaMemcpy (meta->out_buf_ptrs_host[0], meta->out_buf_ptrs_dev[0],
            info->inferDims.numElements * 4, cudaMemcpyDeviceToHost);
        }

        NvDsInferDimsCHW dims;
        getDimsCHWFromDims (dims, meta->output_layers_info[0].inferDims);
        unsigned int emb_dim = dims.c;
        float *outputCoverageBuffer = (float *) meta->output_layers_info[0].buffer;

        std::vector<float> emb(outputCoverageBuffer, outputCoverageBuffer + emb_dim);
        data::Face face;
        face.set_batch(1);
        face.set_emb_dim(512);
        *face.mutable_emb() = {emb.begin(), emb.end()};
        std::string binary;
        std::string binary_b64;
        face.SerializeToString(&binary);
        binary_b64 = base64_encode(binary);
        // std::cout << binary_b64 << std::endl;

        // gchar *pointer = g_strdup (binary_b64.c_str());
        // std::cout << pointer << std::endl;
        NvDsEventMsgMeta *msg_meta = (NvDsEventMsgMeta *) g_malloc0 (sizeof (NvDsEventMsgMeta));
        msg_meta->ts = (gchar *) g_malloc0 (MAX_TIME_STAMP_LEN + 1);
        generate_ts_rfc3339(msg_meta->ts, MAX_TIME_STAMP_LEN);
        msg_meta->camID = g_strdup ("box0-0");
        msg_meta->objType = NVDS_OBJECT_TYPE_FACE;
        NvDsFaceObject *obj = (NvDsFaceObject *) g_malloc0 (sizeof (NvDsFaceObject));
        obj->emb = g_strdup (binary_b64.c_str());

        msg_meta->extMsg = obj;
        msg_meta->extMsgSize = sizeof (NvDsFaceObject);

        NvDsUserMeta *user_event_meta = nvds_acquire_user_meta_from_pool (batch_meta);
        if (user_event_meta) {
          user_event_meta->user_meta_data = (void *) msg_meta;
          user_event_meta->base_meta.meta_type = NVDS_EVENT_MSG_META;
          user_event_meta->base_meta.copy_func = (NvDsMetaCopyFunc) meta_copy_func;
          user_event_meta->base_meta.release_func = (NvDsMetaReleaseFunc) meta_free_func;
          nvds_add_user_meta_to_frame(frame_meta, user_event_meta);
        } else {
          g_print ("Error in attaching event meta to buffer\n");
        }
      }
    }
  }
  return GST_PAD_PROBE_OK;
}


static GstPadProbeReturn
osd_sink_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info, gpointer u_data) {
  
  return GST_PAD_PROBE_OK;
}

static gboolean
bus_call (GstBus * bus, GstMessage * msg, gpointer data)
{
  GMainLoop *loop = (GMainLoop *) data;
  switch (GST_MESSAGE_TYPE (msg)) {
    case GST_MESSAGE_EOS:
      g_print ("End of stream\n");
      g_main_loop_quit (loop);
      break;
    case GST_MESSAGE_ERROR:{
      gchar *debug;
      GError *error;
      gst_message_parse_error (msg, &error, &debug);
      g_printerr ("ERROR from element %s: %s\n",
          GST_OBJECT_NAME (msg->src), error->message);
      if (debug)
        g_printerr ("Error details: %s\n", debug);
      g_free (debug);
      g_error_free (error);
      g_main_loop_quit (loop);
      break;
    }
    default:
      break;
  }
  return TRUE;
}

static void
cb_newpad (GstElement * decodebin, GstPad * decoder_src_pad, gpointer data)
{
  g_print ("In cb_newpad\n");
  GstCaps *caps = gst_pad_get_current_caps (decoder_src_pad);
  const GstStructure *str = gst_caps_get_structure (caps, 0);
  const gchar *name = gst_structure_get_name (str);
  GstElement *source_bin = (GstElement *) data;
  GstCapsFeatures *features = gst_caps_get_features (caps, 0);

  /* Need to check if the pad created by the decodebin is for video and not
   * audio. */
  if (!strncmp (name, "video", 5)) {
    /* Link the decodebin pad only if decodebin has picked nvidia
     * decoder plugin nvdec_*. We do this by checking if the pad caps contain
     * NVMM memory features. */
    if (gst_caps_features_contains (features, MEMORY_FEATURES)) {
      /* Get the source bin ghost pad */
      GstPad *bin_ghost_pad = gst_element_get_static_pad (source_bin, "src");
      if (!gst_ghost_pad_set_target (GST_GHOST_PAD (bin_ghost_pad),
              decoder_src_pad)) {
        g_printerr ("Failed to link decoder src pad to source bin ghost pad\n");
      }
      gst_object_unref (bin_ghost_pad);
    } else {
      g_printerr ("Error: Decodebin did not pick nvidia decoder plugin.\n");
    }
  }
}

static void
decodebin_child_added (GstChildProxy * child_proxy, GObject * object,
    gchar * name, gpointer user_data)
{
  g_print ("Decodebin child added: %s\n", name);
  if (g_strrstr (name, "decodebin") == name) {
    g_signal_connect (G_OBJECT (object), "child-added",
        G_CALLBACK (decodebin_child_added), user_data);
  }
}

static GstElement *
create_source_bin (guint index, gchar * uri)
{
  GstElement *bin = NULL, *uri_decode_bin = NULL;
  gchar bin_name[16] = { };

  g_snprintf (bin_name, 15, "source-bin-%02d", index);
  /* Create a source GstBin to abstract this bin's content from the rest of the
   * pipeline */
  bin = gst_bin_new (bin_name);

  /* Source element for reading from the uri.
   * We will use decodebin and let it figure out the container format of the
   * stream and the codec and plug the appropriate demux and decode plugins. */
  uri_decode_bin = gst_element_factory_make ("uridecodebin", "uri-decode-bin");

  if (!bin || !uri_decode_bin) {
    g_printerr ("1:One element in source bin could not be created.\n");
    return NULL;
  }

  /* We set the input uri to the source element */
  g_object_set (G_OBJECT (uri_decode_bin), "uri", uri, NULL);

  /* Connect to the "pad-added" signal of the decodebin which generates a
   * callback once a new pad for raw data has beed created by the decodebin */
  g_signal_connect (G_OBJECT (uri_decode_bin), "pad-added",
      G_CALLBACK (cb_newpad), bin);
  g_signal_connect (G_OBJECT (uri_decode_bin), "child-added",
      G_CALLBACK (decodebin_child_added), bin);

  gst_bin_add (GST_BIN (bin), uri_decode_bin);

  /* We need to create a ghost pad for the source bin which will act as a proxy
   * for the video decoder src pad. The ghost pad will not have a target right
   * now. Once the decode bin creates the video decoder and generates the
   * cb_newpad callback, we will set the ghost pad target to the video decoder
   * src pad. */
  if (!gst_element_add_pad (bin, gst_ghost_pad_new_no_target ("src",
              GST_PAD_SRC))) {
    g_printerr ("Failed to add ghost pad in source bin\n");
    return NULL;
  }

  return bin;
}

int main(int argc, char *argv[]) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  GMainLoop *loop = NULL;
  GstElement *pipeline = NULL, *streammux = NULL, *sink = NULL, *pgie =
      NULL, *nvvidconv = NULL, *caps_filter = NULL, *nvosd = NULL,
      *queue = NULL, *nvtracker = NULL, *sgie = NULL, *tee = NULL, *queue1 = NULL, *queue2 = NULL, 
      *msgconv = NULL, *msgbroker = NULL;

#ifdef PLATFORM_TEGRA
  GstElement *transform = NULL;
#endif
  GstBus *bus = NULL;
  guint bus_watch_id;
  GstPad *pgie_src_pad = NULL, *sgie_src_pad = NULL;

  /* Check input arguments */
  if (argc != 2) {
    g_printerr ("Usage: %s <uri>\n", argv[0]);
    return -1;
  }

  /* Standard GStreamer initialization */
  gst_init (&argc, &argv);
  loop = g_main_loop_new (NULL, FALSE);

  /* Create gstreamer elements */
  /* Create Pipeline element that will form a connection of other elements */
  pipeline = gst_pipeline_new ("retinaface-pipeline");

  /* Create nvstreammux instance to form batches from one or more sources. */
  streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");

  if (!pipeline || !streammux) {
    g_printerr ("2:One element could not be created. Exiting.\n");
    return -1;
  }
  gst_bin_add (GST_BIN (pipeline), streammux);

  GstPad *sinkpad, *srcpad, *osd_sink_pad, *tee_render_pad, *tee_msg_pad;
  gchar pad_name_sink[16] = "sink_0";
  gchar pad_name_src[16] = "src";

  GstElement *source_bin = create_source_bin (0, argv[1]);

  if (!source_bin) {
    g_printerr ("Failed to create source bin. Exiting.\n");
    return -1;
  }

  gst_bin_add (GST_BIN (pipeline), source_bin);

  sinkpad = gst_element_get_request_pad (streammux, pad_name_sink);
  if (!sinkpad) {
    g_printerr ("Streammux request sink pad failed. Exiting.\n");
    return -1;
  }

  srcpad = gst_element_get_static_pad (source_bin, pad_name_src);
  if (!srcpad) {
    g_printerr ("Failed to get src pad of source bin. Exiting.\n");
    return -1;
  }

  if (gst_pad_link (srcpad, sinkpad) != GST_PAD_LINK_OK) {
    g_printerr ("Failed to link source bin to stream muxer. Exiting.\n");
    return -1;
  }

  gst_object_unref (srcpad);
  gst_object_unref (sinkpad);

  /* Use nvinfer to run inferencing on decoder's output,
   * behaviour of inferencing is set through config file */
  pgie = gst_element_factory_make ("nvinfer", "primary-nvinference-engine");

  /* We need to have a tracker to track the identified objects */
  nvtracker = gst_element_factory_make ("nvtracker", "tracker");

  queue = gst_element_factory_make ("queue", NULL);
  queue1 = gst_element_factory_make ("queue", "nvtee-que1");
  queue2 = gst_element_factory_make ("queue", "nvtee-que2");

  sgie = gst_element_factory_make ("nvinfer", "secondary-nvinference-engine");

  /* Use convertor to convert from NV12 to RGBA as required by dsexample */
  nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter");

  caps_filter = gst_element_factory_make ("capsfilter", NULL);

  /* Create OSD to draw on the converted RGBA buffer */
  nvosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");

  /* Create msg converter to generate payload from buffer metadata */
  msgconv = gst_element_factory_make ("nvmsgconv", "nvmsg-converter");

  /* Create msg broker to send payload to server */
  msgbroker = gst_element_factory_make ("nvmsgbroker", "nvmsg-broker");

  tee = gst_element_factory_make ("tee", "nvsink-tee");

  /* Finally render the osd output */
#ifdef PLATFORM_TEGRA
  transform = gst_element_factory_make ("nvegltransform", "nvegl-transform");
#endif
  sink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");

  if (!pgie || !nvtracker || !queue || !queue1 || !queue2 || !sgie || !nvvidconv || !caps_filter || !nvosd  || !msgconv || !msgbroker || !tee || !sink) {
    g_printerr ("3:One element could not be created. Exiting.\n");
    return -1;
  }
#ifdef PLATFORM_TEGRA
  if (!transform) {
    g_printerr ("One tegra element could not be created. Exiting.\n");
    return -1;
  }
#endif

  g_object_set (G_OBJECT (streammux), "width", MUXER_OUTPUT_WIDTH, "height",
    MUXER_OUTPUT_HEIGHT, "batch-size", 1,
    "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);
  
  /* Set all the necessary properties of the nvinfer element,
   * the necessary ones are : */
  g_object_set (G_OBJECT (pgie), "config-file-path", INFER_PGIE_CONFIG_FILE, NULL);

  g_object_set (G_OBJECT (sgie), "config-file-path", INFER_SGIE1_CONFIG_FILE,
        "output-tensor-meta", TRUE, "process-mode", 2, NULL);
  
  /* Set necessary properties of the tracker element. */
  if (!set_tracker_properties(nvtracker)) {
    g_printerr ("Failed to set tracker properties. Exiting.\n");
    return -1;
  }
  
  g_object_set (G_OBJECT(msgconv), "config", MSCONV_CONFIG_FILE, NULL);
  g_object_set (G_OBJECT(msgconv), "payload-type", schema_type, NULL);
  g_object_set (G_OBJECT(msgconv), "msg2p-lib", msg2p_lib, NULL);

  g_object_set (G_OBJECT(msgbroker), "proto-lib", proto_lib,
                "conn-str", conn_str, "sync", FALSE, NULL);
  if (topic) {
    g_object_set (G_OBJECT(msgbroker), "topic", topic, NULL);
  }

  if (cfg_file) {
    g_object_set (G_OBJECT(msgbroker), "config", cfg_file, NULL);
  }


#ifndef PLATFORM_TEGRA
  /* Set properties of the nvvideoconvert element
   * requires unified cuda memory for opencv blurring on CPU
   */
  g_object_set (G_OBJECT (nvvidconv), "nvbuf-memory-type", 3, NULL);
#endif

  /* Set properties of the caps_filter element */
  GstCaps *caps =
      gst_caps_new_simple ("video/x-raw", "format", G_TYPE_STRING, "RGBA",
      NULL);
  GstCapsFeatures *feature = gst_caps_features_new (MEMORY_FEATURES, NULL);
  gst_caps_set_features (caps, 0, feature);

  g_object_set (G_OBJECT (caps_filter), "caps", caps, NULL);

  /* Set properties of the sink element */
  g_object_set (G_OBJECT (sink), "sync", FALSE, NULL);

  /* we add a message handler */
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);

  /* Set up the pipeline */
  /* we add all elements into the pipeline */
  gst_bin_add_many (GST_BIN (pipeline), nvvidconv, caps_filter, pgie, nvtracker, queue, sgie, nvosd, msgconv, msgbroker, tee, queue1, queue2, NULL);

#ifdef PLATFORM_TEGRA
  gst_bin_add_many (GST_BIN (pipeline), transform, sink, NULL);
#else
  gst_bin_add_many (GST_BIN (pipeline), sink, NULL);
#endif

#ifdef PLATFORM_TEGRA
  if (!gst_element_link_many (streammux, pgie, nvvidconv,
          caps_filter, nvosd, transform, sink, NULL)) {
    g_printerr ("Elements could not be linked: 2. Exiting.\n");
    return -1;
  }
#else
  if (!gst_element_link_many (streammux, nvvidconv, caps_filter, pgie, nvtracker, queue, sgie, nvosd, tee, NULL)) {
    g_printerr ("Elements could not be linked: 2. Exiting.\n");
    return -1;
  }
#endif

  if (!gst_element_link_many (queue1, msgconv, msgbroker, NULL)) {
    g_printerr ("Elements could not be linked. Exiting.\n");
    return -1;
  }

  if (!gst_element_link (queue2, sink)) {
    g_printerr ("Elements could not be linked. Exiting.\n");
    return -1;
  }

  tee_msg_pad = gst_element_get_request_pad (tee, "src_%u");
  tee_render_pad = gst_element_get_request_pad (tee, "src_%u");
  sinkpad = gst_element_get_static_pad (queue1, "sink");
  if (!tee_msg_pad || !tee_render_pad) {
    g_printerr ("Unable to get request pads\n");
    return -1;
  }

  if (gst_pad_link (tee_msg_pad, sinkpad) != GST_PAD_LINK_OK) {
    g_printerr ("Unable to link tee and message converter\n");
    gst_object_unref (sinkpad);
    return -1;
  }

  gst_object_unref (sinkpad);

  sinkpad = gst_element_get_static_pad (queue2, "sink");
  if (gst_pad_link (tee_render_pad, sinkpad) != GST_PAD_LINK_OK) {
    g_printerr ("Unable to link tee and render\n");
    gst_object_unref (sinkpad);
    return -1;
  }

  gst_object_unref (sinkpad);

  /* Lets add probe to get informed of the meta data generated, we add probe to
   * the sink pad of the osd element, since by that time, the buffer would have
   * had got all the metadata. */
  osd_sink_pad = gst_element_get_static_pad (nvosd, "sink");
  if (!osd_sink_pad)
    g_print ("Unable to get sink pad\n");
  else
    gst_pad_add_probe (osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
        osd_sink_pad_buffer_probe, NULL, NULL);
  gst_object_unref (osd_sink_pad);

  /* Add probe to get informed of the meta data generated, we add probe to
   * the source pad of PGIE's next queue element, since by that time, PGIE's
   * buffer would have had got tensor metadata. */
  pgie_src_pad = gst_element_get_static_pad (pgie, "src");
  gst_pad_add_probe (pgie_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
      pgie_pad_buffer_probe, NULL, NULL);
  
  sgie_src_pad = gst_element_get_static_pad (sgie, "src");
  gst_pad_add_probe (sgie_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
      sgie_pad_buffer_probe, NULL, NULL);

  GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN(pipeline), GST_DEBUG_GRAPH_SHOW_ALL, "retinaface-pipeline");

  /* Set the pipeline to "playing" state */
  g_print ("Now playing: %s\n", argv[1]);
  gst_element_set_state (pipeline, GST_STATE_PLAYING);

  /* Wait till pipeline encounters an error or EOS */
  g_print ("Running...\n");
  g_main_loop_run (loop);

  /* Out of the main loop, clean up nicely */
  g_print ("Returned, stopping playback\n");
  gst_element_set_state (pipeline, GST_STATE_NULL);
  g_print ("Deleting pipeline\n");
  gst_object_unref (GST_OBJECT (pipeline));
  g_source_remove (bus_watch_id);
  g_main_loop_unref (loop);

  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}
