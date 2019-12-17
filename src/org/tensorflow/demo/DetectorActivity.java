/*
 * Copyright 2016 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.demo;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Message;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.view.Display;
import android.view.Gravity;
import android.view.Surface;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.net.Socket;
import java.util.LinkedList;
import java.util.List;
import java.util.Vector;
import org.tensorflow.demo.Store;
import org.tensorflow.demo.OverlayView.DrawCallback;
import org.tensorflow.demo.env.BorderedText;
import org.tensorflow.demo.env.ImageUtils;
import org.tensorflow.demo.env.Logger;
import org.tensorflow.demo.tracking.MultiBoxTracker;
import org.tensorflow.demo.tracking.MultiBoxTracker.TrackedRecognition;
import org.tensorflow.demo.R; // Explicit import needed for internal Google builds.
import org.w3c.dom.Text;

import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.RequestBody;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import retrofit2.Retrofit;

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();

  // Configuration values for the prepackaged multibox model.
  private static final int MB_INPUT_SIZE = 224;
  private static final int MB_IMAGE_MEAN = 128;
  private static final float MB_IMAGE_STD = 128;
  private static final String MB_INPUT_NAME = "ResizeBilinear";
  private static final String MB_OUTPUT_LOCATIONS_NAME = "output_locations/Reshape";
  private static final String MB_OUTPUT_SCORES_NAME = "output_scores/Reshape";
  private static final String MB_MODEL_FILE = "file:///android_asset/multibox_model.pb";
  private static final String MB_LOCATION_FILE =
      "file:///android_asset/multibox_location_priors.txt";

  private static final int TF_OD_API_INPUT_SIZE = 300;
  private static final String TF_OD_API_MODEL_FILE =
      "file:///android_asset/ssd_mobilenet_v1_android_export.pb";
  private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/coco_labels_list.txt";

  // Configuration values for tiny-yolo-voc. Note that the graph is not included with TensorFlow and
  // must be manually placed in the assets/ directory by the user.
  // Graphs and models downloaded from http://pjreddie.com/darknet/yolo/ may be converted e.g. via
  // DarkFlow (https://github.com/thtrieu/darkflow). Sample command:
  // ./flow --model cfg/tiny-yolo-voc.cfg --load bin/tiny-yolo-voc.weights --savepb --verbalise
  private static final String YOLO_MODEL_FILE = "file:///android_asset/graph-tiny-yolo-voc.pb";
  private static final int YOLO_INPUT_SIZE = 416;
  private static final String YOLO_INPUT_NAME = "input";
  private static final String YOLO_OUTPUT_NAMES = "output";
  private static final int YOLO_BLOCK_SIZE = 32;

  // Which detection model to use: by default uses Tensorflow Object Detection API frozen
  // checkpoints.  Optionally use legacy Multibox (trained using an older version of the API)
  // or YOLO.
  private enum DetectorMode {
    TF_OD_API, MULTIBOX, YOLO;
  }
  private static final DetectorMode MODE = DetectorMode.TF_OD_API;

  // Minimum detection confidence to track a detection.
  private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.6f;
  private static final float MINIMUM_CONFIDENCE_MULTIBOX = 0.1f;
  private static final float MINIMUM_CONFIDENCE_YOLO = 0.25f;

  private static final boolean MAINTAIN_ASPECT = MODE == DetectorMode.YOLO;

  private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);

  private static final boolean SAVE_PREVIEW_BITMAP = false;
  private static final float TEXT_SIZE_DIP = 10;

  private Integer sensorOrientation;

  private Classifier detector;

  private long lastProcessingTimeMs;
  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;
  private Bitmap cropCopyBitmap = null;

  private boolean computingDetection = false;

  private long timestamp = 0;

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;

  private MultiBoxTracker tracker;

  private byte[] luminanceCopy;

  private BorderedText borderedText;

//  private Socket socket;
  // 获取图片的大小和旋转角度？进行一些初始化的东西
  // frame是原始版本，crop是削减版本？使用frameToCrop矩阵可以进行转换，crop版本的尺寸是预设好的
  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {
    final float textSizePx =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

    tracker = new MultiBoxTracker(this);

    int cropSize = TF_OD_API_INPUT_SIZE;
    if (MODE == DetectorMode.YOLO) {
      detector =
          TensorFlowYoloDetector.create(
              getAssets(),
              YOLO_MODEL_FILE,
              YOLO_INPUT_SIZE,
              YOLO_INPUT_NAME,
              YOLO_OUTPUT_NAMES,
              YOLO_BLOCK_SIZE);
      cropSize = YOLO_INPUT_SIZE;
    } else if (MODE == DetectorMode.MULTIBOX) {
      detector =
          TensorFlowMultiBoxDetector.create(
              getAssets(),
              MB_MODEL_FILE,
              MB_LOCATION_FILE,
              MB_IMAGE_MEAN,
              MB_IMAGE_STD,
              MB_INPUT_NAME,
              MB_OUTPUT_LOCATIONS_NAME,
              MB_OUTPUT_SCORES_NAME);
      cropSize = MB_INPUT_SIZE;
    } else {
      try {
        detector = TensorFlowObjectDetectionAPIModel.create(
            getAssets(), TF_OD_API_MODEL_FILE, TF_OD_API_LABELS_FILE, TF_OD_API_INPUT_SIZE);
        cropSize = TF_OD_API_INPUT_SIZE;
      } catch (final IOException e) {
        LOGGER.e(e, "Exception initializing classifier!");
        Toast toast =
            Toast.makeText(
                getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
        toast.show();
        finish();
      }
    }

    previewWidth = size.getWidth();
    previewHeight = size.getHeight();

    sensorOrientation = rotation - getScreenOrientation();
    LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
    croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

    frameToCropTransform =
        ImageUtils.getTransformationMatrix(
            previewWidth, previewHeight,
            cropSize, cropSize,
            sensorOrientation, MAINTAIN_ASPECT);

    cropToFrameTransform = new Matrix();
    frameToCropTransform.invert(cropToFrameTransform);

      switchModeButton = (Button) findViewById(R.id.switch_mode_button);
      modeText = (TextView) findViewById(R.id.mode_text);
      switchWayButton = (Button) findViewById(R.id.switch_way_button);
      wayText = (TextView) findViewById(R.id.way_text);

      switchModeButton.setOnClickListener(new View.OnClickListener() {
        @Override
        public void onClick(View v) {
            Store.setWhetherLocal(!Store.isWhetherLocal());
            if (Store.isWhetherLocal()) {
                modeText.setText("当前检测模式：本地");
            } else {
                modeText.setText("当前检测模式：远程");
            }
        }
    });

      switchWayButton.setOnClickListener(new View.OnClickListener() {
          @Override
          public void onClick(View v) {
              Store.setWhetherAuto(!Store.isWhetherLocal());
              if (Store.isWhetherAuto()) {
                  wayText.setText("当前切换方式：自动");
              } else {
                  wayText.setText("当前切换方式：手动");
              }
          }
      });

    trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
    trackingOverlay.addCallback(
        new DrawCallback() {
          @Override
          public void drawCallback(final Canvas canvas) {
            tracker.draw(canvas);
            if (isDebug()) {
              tracker.drawDebug(canvas);
            }
          }
        });

    addCallback(
        new DrawCallback() {
          @Override
          public void drawCallback(final Canvas canvas) {
            if (!isDebug()) {
              return;
            }
            final Bitmap copy = cropCopyBitmap;
            if (copy == null) {
              return;
            }

            final int backgroundColor = Color.argb(100, 0, 0, 0);
            canvas.drawColor(backgroundColor);

            final Matrix matrix = new Matrix();
            final float scaleFactor = 2;
            matrix.postScale(scaleFactor, scaleFactor);
            matrix.postTranslate(
                canvas.getWidth() - copy.getWidth() * scaleFactor,
                canvas.getHeight() - copy.getHeight() * scaleFactor);
            canvas.drawBitmap(copy, matrix, new Paint());

            final Vector<String> lines = new Vector<String>();
            if (detector != null) {
              final String statString = detector.getStatString();
              final String[] statLines = statString.split("\n");
              for (final String line : statLines) {
                lines.add(line);
              }
            }
            lines.add("");

            lines.add("Frame: " + previewWidth + "x" + previewHeight);
            lines.add("Crop: " + copy.getWidth() + "x" + copy.getHeight());
            lines.add("View: " + canvas.getWidth() + "x" + canvas.getHeight());
            lines.add("Rotation: " + sensorOrientation);
            lines.add("Inference time: " + lastProcessingTimeMs + "ms");

            borderedText.drawLines(canvas, 10, canvas.getHeight() - 10, lines);
          }
        });
  }

  OverlayView trackingOverlay;
  Button switchModeButton;
  Button switchWayButton;
  TextView modeText;
  TextView wayText;

  // 对每张图片的操作，getLuminance获得图片的bytes[],getRgbBytes()获得rgb的bitmap
  @Override
  protected void processImage() {
    if (Store.isWhetherLocal() == false) {
        computingDetection = false;
    }
    LOGGER.i("开始processImage");
    ++timestamp;
    final long currTimestamp = timestamp;
    final byte[] originalLuminance = getLuminance();
    tracker.onFrame(
            previewWidth,
            previewHeight,
            getLuminanceStride(),
            sensorOrientation,
            originalLuminance,
            timestamp);
    trackingOverlay.postInvalidate();

    // No mutex needed as this method is not reentrant.
    if (computingDetection) {
      readyForNextImage();
      LOGGER.i("本地模式正在计算，帧丢弃");
      return;
    }
    computingDetection = true;
    LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

    // 由拍摄到的rgbFrame写一个rgbbitmap
    rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

    if (luminanceCopy == null) {
      luminanceCopy = new byte[originalLuminance.length];
    }
    System.arraycopy(originalLuminance, 0, luminanceCopy, 0, originalLuminance.length);
    // 之前的图片不做了准备好开始下一张？
    readyForNextImage();

    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
    // For examining the actual TF input.
    if (SAVE_PREVIEW_BITMAP) {
      ImageUtils.saveBitmap(croppedBitmap);
    }

    if (Store.isWhetherLocal()) {
        // 本地模式这里开始做处理
        runInBackground(
                new Runnable() {
                    @Override
                    public void run() {
                        LOGGER.i("Running detection on image " + currTimestamp);
                        // 这里应该是对图片进行不同object的检测和分割，具体识别还在后面
                        final long startTime = SystemClock.uptimeMillis();
                        final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);
                        lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

                        cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
                        final Canvas canvas = new Canvas(cropCopyBitmap);
                        final Paint paint = new Paint();
                        paint.setColor(Color.RED);
                        paint.setStyle(Style.STROKE);
                        paint.setStrokeWidth(2.0f);

                        // 要求的最小的自信度NDD
                        float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                        switch (MODE) {
                            case TF_OD_API:
                                minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                                break;
                            case MULTIBOX:
                                minimumConfidence = MINIMUM_CONFIDENCE_MULTIBOX;
                                break;
                            case YOLO:
                                minimumConfidence = MINIMUM_CONFIDENCE_YOLO;
                                break;
                        }

                        final List<Classifier.Recognition> mappedRecognitions =
                                new LinkedList<Classifier.Recognition>();

                        // 在crop版本的canvas上画出来， 并将转换到frame版本的结果存储在mappedRecognitions中
                        for (final Classifier.Recognition result : results) {
                            final RectF location = result.getLocation();
                            if (location != null && result.getConfidence() >= minimumConfidence) {
                                canvas.drawRect(location, paint);

                                cropToFrameTransform.mapRect(location);
                                result.setLocation(location);
                                mappedRecognitions.add(result);
                            }
                        }

                        // 对在原始版本上的各种object进行识别
                        tracker.trackResults(mappedRecognitions, luminanceCopy, currTimestamp);
                        trackingOverlay.postInvalidate();

                        requestRender();
                        computingDetection = false;
                    }
                });
    } else {
        LOGGER.i("进入远程模式");
        if (currTimestamp % 10 ==0) {
            LOGGER.i("帧满足条件，当前帧数："+currTimestamp);
            Store.setIsComputing(true);

            new Thread() {
                public void run() {
                    try {
                        // 传送图片
                        final ByteArrayOutputStream outputStream = new ByteArrayOutputStream(rgbFrameBitmap.getByteCount());
                        rgbFrameBitmap.compress(Bitmap.CompressFormat.PNG, 100, outputStream);

                        LOGGER.i("开始新线程发送socket");
                        Thread.sleep(3000);

//                                  Socket socket = new Socket(Store.ip, Store.port);
//                                  DataOutputStream socketOutputStream = new DataOutputStream(socket.getOutputStream());
//                                  DataInputStream socketInputStream = new DataInputStream(socket.getInputStream());
//
//                                  int size = outputStream.toByteArray().length;
//                                  String sizeStr = String.valueOf(size);
////                          String sizeStrr = new String(sizeStr.getBytes(), "UTF-8");
//                                  socketOutputStream.write(sizeStr.getBytes("UTF-8"));
//                                  LOGGER.i("pppppppp"+outputStream.size());
//                                  LOGGER.i("pppppppp"+sizeStr);
//                                  LOGGER.i("pppppppp"+sizeStr.getBytes("UTF-8"));
//                                  LOGGER.i("pppppppp"+sizeStr.getBytes().length);
//
//                                  socketOutputStream.write(outputStream.toByteArray());
//                                  socketOutputStream.flush();
//
////                          byte[] greets = new byte[1024];
//                                  byte[] response = new byte[4096];
////                          socketInputStream.read(greets);
//                                  socketInputStream.read(response);
////                          String greeting = new String(greets);
//                                  String resultStr = new String(response);
//                                  String[] results = resultStr.trim().split(",");
//                                  socket.close();
//
//                                  LOGGER.i("socket收回的结果为"+resultStr);
//
//                                  Integer resultNum = Integer.valueOf(results[0]);
//
//                                  List<TrackedRecognition> trackedRecognitions = new LinkedList<TrackedRecognition>();
//
//                                  int i = 0;
//                                  for (; i < resultNum; i++) {
//                                      Float left = Float.valueOf(results[i * 6 + 1]);
//                                      Float top = Float.valueOf(results[i * 6 + 2]);
//                                      Float right = Float.valueOf(results[i * 6 + 3]);
//                                      Float bottom = Float.valueOf(results[i * 6 + 4]);
//                                      Float confidence = Float.valueOf(results[i * 6 + 5]);
//                                      String label = results[i * 6 + 6];
//                                      LOGGER.i("新增一个框为"+left+" "+top+" "+right+" "+bottom+" "+confidence+" "+label);
//                                      RectF rectF = new RectF(left, top, right, bottom);
//                                      final TrackedRecognition trackedRecognition = new TrackedRecognition();
//                                      trackedRecognition.location = rectF;
//                                      trackedRecognition.detectionConfidence = confidence;
//                                      int index=(int)(Math.random()*tracker.COLORS.length);
//                                      trackedRecognition.color = tracker.COLORS[index];
//                                      trackedRecognition.title = label;
//                                      trackedRecognitions.add(trackedRecognition);
//                                  }
//                                  if (!Store.isDrawing) {
//                                      tracker.setTrackedObjects(trackedRecognitions);
//                                  }
                        LOGGER.i("socket线程即将结束，当前帧数为"+currTimestamp);

                        trackingOverlay.postInvalidate();
                        requestRender();
                        Store.setIsComputing(false);

                    }
                    catch (Exception e) {
                        e.printStackTrace();
                    }
                };
            }.start();
        } else {
            LOGGER.i("帧不满足条件，当前帧数:"+currTimestamp);
        }

        trackingOverlay.postInvalidate();
        requestRender();
        computingDetection = false;
    }
  }

  @Override
  protected int getLayoutId() {
    return R.layout.camera_connection_fragment_tracking;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

  @Override
  public void onSetDebug(final boolean debug) {
    detector.enableStatLogging(debug);
  }

    Bitmap adjustPhotoRotation(Bitmap bitmap, int orientationDegree) {



        Matrix matrix = new Matrix();
        matrix.setRotate(orientationDegree, (float) bitmap.getWidth() / 2,
                (float) bitmap.getHeight() / 2);
        float targetX, targetY;
        if (orientationDegree == 90) {
            targetX = bitmap.getHeight();
            targetY = 0;
        } else {
            targetX = bitmap.getHeight();
            targetY = bitmap.getWidth();
        }


        final float[] values = new float[9];
        matrix.getValues(values);


        float x1 = values[Matrix.MTRANS_X];
        float y1 = values[Matrix.MTRANS_Y];


        matrix.postTranslate(targetX - x1, targetY - y1);


        Bitmap canvasBitmap = Bitmap.createBitmap(bitmap.getHeight(), bitmap.getWidth(),
                Bitmap.Config.ARGB_8888);


        Paint paint = new Paint();
        Canvas canvas = new Canvas(canvasBitmap);
        canvas.drawBitmap(bitmap, matrix, paint);


        return canvasBitmap;
    }

    void sendHttp() {
        Store.setIsComputing(true);
        final ByteArrayOutputStream outputStream = new ByteArrayOutputStream(rgbFrameBitmap.getByteCount());
        rgbFrameBitmap.compress(Bitmap.CompressFormat.PNG, 100, outputStream);

        final long startTime = System.currentTimeMillis();
        RequestBody requestBody = RequestBody.create(MediaType.parse("image/png"), outputStream.toByteArray());
        MultipartBody.Part picRgb = MultipartBody.Part.createFormData("pic_rgb", "rgb", requestBody);
        Retrofit retrofit = Store.getRetrofit();
        PictureApi pictureApi = retrofit.create(PictureApi.class);
        pictureApi.uploadImage(picRgb).enqueue(new Callback<String>() {
            @Override
            public void onResponse(Call<String> call, Response<String> response) {
                try {
                    long endTime = System.currentTimeMillis();
                    LOGGER.i("传输来回时间为" + (endTime - startTime) + "毫秒");
                    LOGGER.i(response.body());
                    String responseStr = response.body();
                    String[] results = responseStr.trim().split(",");
                    Integer resultNum = Integer.valueOf(results[0]);

                    List<TrackedRecognition> trackedRecognitions = new LinkedList<TrackedRecognition>();

                    int i = 0;
                    for (; i < resultNum; i++) {
                        Float left = Float.valueOf(results[i * 6 + 1]);
                        Float top = Float.valueOf(results[i * 6 + 2]);
                        Float right = Float.valueOf(results[i * 6 + 3]);
                        Float bottom = Float.valueOf(results[i * 6 + 4]);
                        Float confidence = Float.valueOf(results[i * 6 + 5]);
                        String label = results[i * 6 + 6];
                        LOGGER.i("添加一个框为"+left+" "+top+" "+right+" "+bottom+" "+confidence+" "+label);
                        RectF rectF = new RectF(left, top, right, bottom);
                        final TrackedRecognition trackedRecognition = new TrackedRecognition();
                        trackedRecognition.location = rectF;
                        trackedRecognition.detectionConfidence = confidence;
                        int index=(int)(Math.random()*tracker.COLORS.length);
                        trackedRecognition.color = tracker.COLORS[index];
                        trackedRecognition.title = label;
                        trackedRecognitions.add(trackedRecognition);
                    }
                    tracker.setTrackedObjects(trackedRecognitions);

                    trackingOverlay.postInvalidate();

                    requestRender();
                    Store.setIsComputing(false);

                } catch (Exception e) {
                    LOGGER.i("http出错了");
                    e.printStackTrace();
                }
            }

            @Override
            public void onFailure(Call<String> call, Throwable t) {
                t.printStackTrace();
            }
        });
    }
}