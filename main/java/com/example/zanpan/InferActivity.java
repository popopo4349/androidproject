package com.example.zanpan;


import android.graphics.Bitmap;
import androidx.appcompat.app.AppCompatActivity;

import android.net.Uri;
import android.os.Bundle;
import android.widget.ImageView;

public class InferActivity extends AppCompatActivity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_infer);

        // 画像のURIを受け取る
        String imageUriString = getIntent().getStringExtra("imageUri");
        Uri imageUri = Uri.parse(imageUriString);
//        Bitmap bitmap = getIntent().getParcelableExtra("bitmap");


        // 画像を表示するImageViewにURIをセットする
        ImageView imageView = findViewById(R.id.image_view);
        imageView.setImageURI(imageUri);
    }

}
