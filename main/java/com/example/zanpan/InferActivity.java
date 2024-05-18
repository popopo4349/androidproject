package com.example.zanpan;


import android.content.ContentValues;
import android.content.DialogInterface;
import android.content.Intent;
import android.graphics.Bitmap;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;

import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.widget.Button;
import android.widget.ImageView;
import android.content.DialogInterface;

public class InferActivity extends BaseActivity {
    final String PAGE_NAME = "推論モード";
    private Class<?> activityClass;
    Uri imageUri;
    private ImageView imageView; // メンバ変数として定義

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_infer);
        setupHeader(PAGE_NAME);

        // 画像のURIを受け取る
        String imageUriString = getIntent().getStringExtra("imageUri");
        imageUri = Uri.parse(imageUriString);

        // 画像を表示するImageViewにURIをセットする
        imageView = findViewById(R.id.image_view); // ここでローカル変数として再度定義しているため修正
        imageView.setImageURI(imageUri);

        Button camera_btn = findViewById(R.id.camera_btn);
        camera_btn.setOnClickListener(v -> {
            showOptionDialog();
        });
    }

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

}
