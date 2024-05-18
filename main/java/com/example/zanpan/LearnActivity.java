package com.example.zanpan;



import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AlertDialog;

import android.annotation.SuppressLint;
import android.os.Bundle;
import android.view.GestureDetector;
import android.view.MotionEvent;
import android.view.ScaleGestureDetector;

import android.content.ContentValues;
import android.content.DialogInterface;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.ScaleGestureDetector;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

public class LearnActivity extends BaseActivity {
    final String PAGE_NAME = "学習モード";
    private Class<?> activityClass;
    Uri imageUri;
    private ImageView imageView;
    private float previousX, previousY;
    private ScaleGestureDetector scaleGestureDetector;
    private float scaleFactor = 1.0f;
    private GestureDetector gestureDetector;

    float startX, startY;
    float offsetX, offsetY;

    float maxX, maxY;

    private float initialTranslationX, initialTranslationY;



    private TextView scaleFactorTextView;
    private TextView w_h_View;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_learn);
        setupHeader(PAGE_NAME);

        // 画像のURIを受け取る
        String imageUriString = getIntent().getStringExtra("imageUri");
        Uri imageUri = Uri.parse(imageUriString);

        // 画像を表示するImageViewにURIをセットする
        imageView = findViewById(R.id.image_view);
        imageView.setImageURI(imageUri);

        imageView.setImageURI(imageUri);
        imageView.setAdjustViewBounds(true);

        scaleGestureDetector = new ScaleGestureDetector(this, new ScaleListener());
        gestureDetector = new GestureDetector(this, new DoubleTapListener());

        scaleFactorTextView = findViewById(R.id.scaleFactorTextView);
        w_h_View = findViewById(R.id.w_h_View);
        Button camera_btn = findViewById(R.id.camera_btn);
        camera_btn.setOnClickListener(v -> {
            showOptionDialog();
        });
    }
/*
再撮影に関する処理
 */
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
/*
zoom機能に関する処理
 */


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
                maxX = (imageView.getWidth() * (imageView.getScaleX()- 1))/2;
                maxY = (imageView.getHeight() * (imageView.getScaleY() - 1))/2;
                break;
            case MotionEvent.ACTION_MOVE:
                // タッチ移動中の座標差を計算して画像ビューの移動量に反映
                float dx = event.getRawX() - startX;
                float dy = event.getRawY() - startY;

                // 移動量を制限して画像ビューの移動量に設定
                float newX = Math.max(Math.min(offsetX + dx, maxX), 0);
                float newY = Math.max(Math.min(offsetY + dy, maxY), 0);
                imageView.setTranslationX(newX);
                imageView.setTranslationY(newY);
                w_h_View.setText("Xmax: " + maxX  + "Ymax" +  maxY + "x" + newX  + "y"  + newY);
                scaleFactorTextView.setText("Scale Factor: " + scaleFactor);
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
            scaleFactor = 1.0f;
            imageView.setScaleX(scaleFactor);
            imageView.setScaleY(scaleFactor);
            return true;
        }
    }
}



