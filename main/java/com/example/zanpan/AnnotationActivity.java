package com.example.zanpan;



import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AlertDialog;
import androidx.core.content.ContextCompat;

import android.annotation.SuppressLint;
import android.content.ContentResolver;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.os.Bundle;
import android.view.GestureDetector;
import android.view.MotionEvent;
import android.view.ScaleGestureDetector;

import android.content.ContentValues;
import android.content.DialogInterface;
import android.content.Intent;
import android.net.Uri;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;

public class AnnotationActivity extends BaseActivity {
    final String PAGE_NAME = "学習モード";
    private Class<?> activityClass;
    Uri imageUri;
    private ImageView imageView;
    private DrawingView drawingView;
    private float previousX, previousY;
    private Button zoom_btn, class_btn, back_btn, remove_btn, upload_btn, camera_btn;
    private ScaleGestureDetector scaleGestureDetector;
    private float scaleFactor = 1.0f;
    private GestureDetector gestureDetector;

    float startX, startY;
    float offsetX, offsetY;
    float maxX, maxY;

    private float initialTranslationX, initialTranslationY;



    private TextView scaleFactorTextView;



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_annotation);
        setupHeader(PAGE_NAME);

        // 画像のURIを受け取る
        String imageUriString = getIntent().getStringExtra("imageUri");
        Uri imageUri = Uri.parse(imageUriString);

        // 画像を表示するImageViewにURIをセットする
        imageView = findViewById(R.id.image_view);
        imageView.setImageURI(imageUri);
        imageView.setAdjustViewBounds(true);

        // drawViewをセットする
        drawingView = findViewById(R.id.drawingView);
        //drawingView.setImageView(imageView);

        //各種ボタンをセットする
        class_btn = findViewById(R.id.class_btn);
        zoom_btn = findViewById(R.id.zoom_btn);
        back_btn = findViewById(R.id.back_btn);
        remove_btn = findViewById(R.id.remove_btn);
        upload_btn = findViewById(R.id.upload_btn);
        camera_btn = findViewById(R.id.camera_btn);

        scaleGestureDetector = new ScaleGestureDetector(this, new ScaleListener());
        gestureDetector = new GestureDetector(this, new DoubleTapListener());

        scaleFactorTextView = findViewById(R.id.scaleFactorTextView);


        class_btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(drawingView.getDrawMode() == true) {
                    showColorDialog();
                }
            }
        });

        back_btn.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                if(drawingView.getDrawMode() == true) {
                    drawingView.setEraserMode();
                }
            }
        });

        remove_btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(drawingView.getDrawMode() == true) {
                    drawingView.clearDrawing();
                }
            }
        });

//        upload_btn.setOnClickListener(new View.OnClickListener() {
//            @Override
//            public void onClick(View v) {
//                // Bitmapを作成して描画内容を取得
//                Bitmap bitmap = Bitmap.createBitmap(drawingView.getWidth(), drawingView.getHeight(), Bitmap.Config.ARGB_8888);
//                Canvas canvas = new Canvas(bitmap);
//                drawingView.draw(canvas);
//                // 画像をアルバムに保存
//                saveImageToAlbum(LearnActivity.this, bitmap, "drawing_" + System.currentTimeMillis());
//            }
//        });
        upload_btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Bitmap canvasBitmap = getCanvasBitmap();
                Intent intent = new Intent(AnnotationActivity.this, UploadActivity.class);
                intent.putExtra("imageUri", imageUriString);
                intent.putExtra("canvasBitmap", bitmapToByteArray(canvasBitmap));
                startActivity(intent);
            }
        });

        camera_btn.setOnClickListener(v -> {
            showOptionDialog();
        });

        zoom_btn.setOnClickListener(v -> {
            drawingView.setDrawMode(!drawingView.getDrawMode());
            if (drawingView.getDrawMode()) {
                class_btn.setBackgroundColor(ContextCompat.getColor(AnnotationActivity.this, R.color.btn_std));
                back_btn.setBackgroundColor(ContextCompat.getColor(AnnotationActivity.this, R.color.btn_std));
                remove_btn.setBackgroundColor(ContextCompat.getColor(AnnotationActivity.this, R.color.btn_std));
            } else {
                class_btn.setBackgroundColor(Color.argb(192, 17, 0, 103));
                back_btn.setBackgroundColor(Color.argb(192, 17, 0, 103));
                remove_btn.setBackgroundColor(Color.argb(192, 17, 0, 103));
            }
        });




    }
    //再撮影による処理
    ActivityResultLauncher<Intent> startForResultTake = registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(),
            result -> {
                if (result.getResultCode() == RESULT_OK) {
                    imageView.setImageURI(imageUri);
                }
            });
    // 内部から読み込んだ画像を次の画面を引き渡す処理
    ActivityResultLauncher<Intent> startForResultLoad = registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(),
            result -> {
                Intent data = result.getData();
                if (data != null && data.getData() != null) {
                    Uri uri = data.getData();
                    imageView.setImageURI(uri);
                }
            }
    );
    //画像をカメラと内部どちらから取得するかの選択ダイアログ
    void showOptionDialog() {
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        final CharSequence[] items = {"カメラから画像を撮影する", "フォルダーから画像を選ぶ"};
        builder.setTitle("選択してください")
                .setItems(items, new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        switch (which) {
                            case 0:
                                takePicture();
                                break;
                            case 1:
                                loadPicture();
                                break;
                        }
                    }
                });
        builder.create().show();
    }
    protected void takePicture() {
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
            ContentValues values = new ContentValues();
            values.put(MediaStore.Images.Media.TITLE, "mypic-" + System.currentTimeMillis() + ".jpg");
            values.put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg");
            imageUri = getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
            takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, imageUri);
            startForResultTake.launch(takePictureIntent);
        }
    }
    //内部フォルダから画像を読み取る処理
    protected void loadPicture() {
        Intent intent = new Intent(Intent.ACTION_PICK, android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startForResultLoad.launch(Intent.createChooser(intent, "Select Picture"));
    }
    //zoom機能に関する処理
    @SuppressLint("ClickableViewAccessibility")
    @Override
    public boolean onTouchEvent(MotionEvent event) {
        scaleGestureDetector.onTouchEvent(event);
        gestureDetector.onTouchEvent(event);

        switch (event.getAction()) {
            case MotionEvent.ACTION_DOWN:
                // タッチ開始時の座標を記録
                startX = event.getRawX();
                startY = event.getRawY();
                offsetX = imageView.getTranslationX();
                offsetY = imageView.getTranslationY();

                // 移動可能な範囲を計算
                maxX = (imageView.getWidth() * (imageView.getScaleX()- 1)/2);
                maxY = (imageView.getHeight() * (imageView.getScaleY() - 1)/2);
                break;
            case MotionEvent.ACTION_MOVE:
                // タッチ移動中の座標差を計算して画像ビューの移動量に反映
                float dx = event.getRawX() - startX;
                float dy = event.getRawY() - startY;

                // 移動量を制限して画像ビューの移動量に設定
                float newX = Math.max(Math.min(offsetX + dx, maxX), -maxX);
                float newY = Math.max(Math.min(offsetY + dy, maxY), -maxY);
                imageView.setTranslationX(newX);
                imageView.setTranslationY(newY);
                drawingView.setTranslationX(newX);
                drawingView.setTranslationY(newY);
//                w_h_View.
                scaleFactorTextView.setText("Xmax: " + maxX  + "Ymax" +  maxY + "x" + offsetX  + "y"  + newY);
                break;
        }
        return true;

    }

    private class ScaleListener extends ScaleGestureDetector.SimpleOnScaleGestureListener {
        @Override
        public boolean onScale(ScaleGestureDetector detector) {

            scaleFactor *= detector.getScaleFactor();

            scaleFactor = Math.max(1.0f, Math.min(scaleFactor, 5.0f));

            imageView.setScaleX(scaleFactor);
            imageView.setScaleY(scaleFactor);
            drawingView.setScaleX(scaleFactor);
            drawingView.setScaleY(scaleFactor);

            scaleFactorTextView.setText("Scale Factor: " + scaleFactor);
            return true;
        }
    }

    private class DoubleTapListener extends GestureDetector.SimpleOnGestureListener {
        @Override
        public boolean onDoubleTap(MotionEvent e) {
            // ダブルタップが検出されたときの処理
            imageView.setTranslationX(initialTranslationX);
            imageView.setTranslationY(initialTranslationY);
            drawingView.setTranslationX(initialTranslationX);
            drawingView.setTranslationY(initialTranslationY);
            scaleFactor = 1.0f;
            imageView.setScaleX(scaleFactor);
            imageView.setScaleY(scaleFactor);
            drawingView.setScaleX(scaleFactor);
            drawingView.setScaleY(scaleFactor);

            return true;
        }
    }

    // アルバムに画像を保存するメソッド
    private void saveImageToAlbum(Context context, Bitmap imageBitmap, String fileName) {
        // 保存する画像の情報を準備する
        ContentValues values = new ContentValues();
        values.put(MediaStore.Images.Media.DISPLAY_NAME, fileName);
        values.put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg");

        // 画像を保存するURIを取得する
        ContentResolver resolver = context.getContentResolver();
        Uri imageUri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);

        // 画像をURIに書き込む
        try {
            OutputStream outputStream = resolver.openOutputStream(imageUri);
            imageBitmap.compress(Bitmap.CompressFormat.JPEG, 100, outputStream);
            outputStream.close();
            Toast.makeText(AnnotationActivity.this, "Saved to album", Toast.LENGTH_SHORT).show();
        } catch (IOException e) {
            e.printStackTrace();
            Toast.makeText(AnnotationActivity.this, "Failed to save", Toast.LENGTH_SHORT).show();
        }
    }
    //クラスを選択するダイアログ
    void showColorDialog() {
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        final CharSequence[] items = {"赤", "青"};
        builder.setTitle("色を選択してください")
                .setItems(items, new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        switch (which) {
                            case 0:
                                drawingView.setPenColor(Color.RED);
                                class_btn.setText("クラス：米");
                                break;
                            case 1:
                                drawingView.setPenColor(Color.BLUE);
                                class_btn.setText("クラス：スープ");
                                break;
                        }
                    }
                });
        builder.create().show();
    }

    private byte[] bitmapToByteArray(Bitmap bitmap) {
        ByteArrayOutputStream stream = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream);
        return stream.toByteArray();
    }

    private Bitmap getCanvasBitmap() {
        Bitmap drawingBitmap = Bitmap.createBitmap(drawingView.getWidth(), drawingView.getHeight(), Bitmap.Config.ARGB_8888);
        Canvas drawingCanvas = new Canvas(drawingBitmap);
        drawingView.draw(drawingCanvas);
        return drawingBitmap;
    }
}



