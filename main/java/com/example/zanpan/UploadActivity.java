package com.example.zanpan;

import static androidx.constraintlayout.helper.widget.MotionEffect.TAG;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.Button;

import androidx.appcompat.app.AppCompatActivity;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.concurrent.TimeUnit;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class UploadActivity extends AppCompatActivity {

    private Button back_btn, upload_btn;
    private static final String TAG = "UploadActivity";
    private static final String SERVER_URL = "http://192.168.0.226:5000/upload";

    private Bitmap bitmap;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_upload); // レイアウトファイルを設定

        ImageView imageView = findViewById(R.id.imageView);
        ImageView canvasView = findViewById(R.id.canvasView);

        back_btn = findViewById(R.id.back_btn); // Buttonの初期化
        upload_btn = findViewById(R.id.up_btn); // upload_btnの初期化

        String imageUriString = getIntent().getStringExtra("imageUri");
        Uri imageUri = Uri.parse(imageUriString);
        imageView.setImageURI(imageUri);

        byte[] byteArray = getIntent().getByteArrayExtra("canvasBitmap");
        Bitmap canvasBitmap = BitmapFactory.decodeByteArray(byteArray, 0, byteArray.length);
        canvasView.setImageBitmap(canvasBitmap);

        back_btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(UploadActivity.this, AnnotationActivity.class);
                intent.putExtra("imageUri", imageUriString);
                startActivity(intent);
            }
        });

        upload_btn.setOnClickListener(new View.OnClickListener() { // upload_btnのクリックリスナー設定
            @Override
            public void onClick(View v) {
                uploadImage(canvasBitmap);

            }
        });
    }


    private void uploadImage(Bitmap bitmap) {
        OkHttpClient client = new OkHttpClient.Builder()
                .connectTimeout(30, TimeUnit.SECONDS)
                .writeTimeout(30, TimeUnit.SECONDS)
                .readTimeout(30, TimeUnit.SECONDS)
                .build();

        // Bitmapをバイト配列に変換
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.JPEG, 100, bos);
        byte[] bitmapdata = bos.toByteArray();

        // バイト配列をRequestBodyに変換
        RequestBody requestBody = RequestBody.create(MediaType.parse("image/jpeg"), bitmapdata);

        // マルチパートのリクエストを作成
        RequestBody multipartBody = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("file", "image.jpg", requestBody)
                .build();

        // リクエストを作成
        Request request = new Request.Builder()
                .url(SERVER_URL)
                .post(multipartBody)
                .build();

        //リクエストを非同期で送信
        client.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(Call call, IOException e) {
                Log.e(TAG, "Failed to upload image", e);
            }

            @Override
            public void onResponse(Call call, Response response) throws IOException {
                if (response.isSuccessful()) {
                    Log.i(TAG, "Image uploaded successfully");
                } else {
                    Log.e(TAG, "Failed to upload image: " + response.message());
                }
            }
        });
    }
}
