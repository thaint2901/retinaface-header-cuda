#include <gst/gst.h>
#include <glib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
/* Open CV headers */
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "nvbufsurface.h"
#include "cuda_runtime_api.h"
#include "nvdsinfer_custom_impl.h"
#include "gstnvdsmeta.h"
#include "gstnvdsinfer.h"
#include "nvds_version.h"

#define INFER_PGIE_CONFIG_FILE  "/nvidia/retinaface-header-cuda/deepstream/infer_config_batch1.txt"

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

extern "C"
  bool NvDsInferParseRetinaNet (std::vector < NvDsInferLayerInfo >
  const &outputLayersInfo, NvDsInferNetworkInfo const &networkInfo,
  NvDsInferParseDetectionParams const &detectionParams,
  std::vector < NvDsInferObjectDetectionInfo > &objectList);

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

    NvBufSurfaceSyncForDevice (surface, frame_meta->batch_id, 0);

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
      for (auto & obj:objectList) {
        NvDsObjectMeta *obj_meta = nvds_acquire_obj_meta_from_pool (batch_meta);
        obj_meta->unique_component_id = meta->unique_id;
        obj_meta->confidence = 0.0;
        /* This is an untracked object. Set tracking_id to -1. */
        obj_meta->object_id = UNTRACKED_OBJECT_ID;
        obj_meta->class_id = 0;

        NvOSD_RectParams & rect_params = obj_meta->rect_params;
        NvOSD_TextParams & text_params = obj_meta->text_params;

        /* Assign bounding box coordinates. */
        rect_params.left = obj.left;
        rect_params.top = obj.top;
        rect_params.width = obj.width;
        rect_params.height = obj.height;

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
  }
  use_device_mem = 1 - use_device_mem;
  gst_buffer_unmap (inbuf, &in_map_info);

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
  GMainLoop *loop = NULL;
  GstElement *pipeline = NULL, *streammux = NULL, *sink = NULL, *pgie =
      NULL, *nvvidconv = NULL, *caps_filter = NULL, *nvosd = NULL,
      *queue = NULL;

#ifdef PLATFORM_TEGRA
  GstElement *transform = NULL;
#endif
  GstBus *bus = NULL;
  guint bus_watch_id;
  GstPad *pgie_src_pad = NULL, *queue_src_pad = NULL;

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

  GstPad *sinkpad, *srcpad;
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

  queue = gst_element_factory_make ("queue", NULL);

  /* Use convertor to convert from NV12 to RGBA as required by dsexample */
  nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter");

  caps_filter = gst_element_factory_make ("capsfilter", NULL);

  /* Create OSD to draw on the converted RGBA buffer */
  nvosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");

  /* Finally render the osd output */
#ifdef PLATFORM_TEGRA
  transform = gst_element_factory_make ("nvegltransform", "nvegl-transform");
#endif
  sink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");

  if (!pgie || !nvvidconv || !caps_filter || !nvosd || !sink) {
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
  gst_bin_add_many (GST_BIN (pipeline), pgie, queue, nvvidconv,
      caps_filter, nvosd, NULL);

#ifdef PLATFORM_TEGRA
  gst_bin_add_many (GST_BIN (pipeline), transform, sink, NULL);
#else
  gst_bin_add_many (GST_BIN (pipeline), sink, NULL);
#endif
  /* we link the elements together */
  /* file-source -> h264-parser -> nvh264-decoder ->
   * nvinfer -> nvvidconv -> nvosd -> video-renderer */
#ifdef PLATFORM_TEGRA
  if (!gst_element_link_many (streammux, pgie, nvvidconv,
          caps_filter, nvosd, transform, sink, NULL)) {
    g_printerr ("Elements could not be linked: 2. Exiting.\n");
    return -1;
  }
#else
  if (!gst_element_link_many (streammux, nvvidconv, caps_filter, pgie, queue, nvosd, sink, NULL)) {
    g_printerr ("Elements could not be linked: 2. Exiting.\n");
    return -1;
  }
#endif

  // /* Lets add probe to get informed of the meta data generated, we add probe to
  //  * the sink pad of the osd element, since by that time, the buffer would have
  //  * had got all the metadata. */
  // osd_sink_pad = gst_element_get_static_pad (nvosd, "sink");
  // if (!osd_sink_pad)
  //   g_print ("Unable to get sink pad\n");
  // else
  //   gst_pad_add_probe (osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
  //       osd_sink_pad_buffer_probe, NULL, NULL);
  // gst_object_unref (osd_sink_pad);

  /* Add probe to get informed of the meta data generated, we add probe to
   * the source pad of PGIE's next queue element, since by that time, PGIE's
   * buffer would have had got tensor metadata. */
  pgie_src_pad = gst_element_get_static_pad (pgie, "src");
  gst_pad_add_probe (pgie_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
      pgie_pad_buffer_probe, NULL, NULL);
  
  queue_src_pad = gst_element_get_static_pad (queue, "src");

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
  return 0;
}
