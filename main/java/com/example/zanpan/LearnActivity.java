package com.example.zanpan;



import androidx.appcompat.app.AppCompatActivity;

import android.net.Uri;
import android.os.Bundle;
import android.widget.ImageView;

public class LearnActivity extends AppCompatActivity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_learn);

        // 画像のURIを受け取る
        String imageUriString = getIntent().getStringExtra("imageUri");
        Uri imageUri = Uri.parse(imageUriString);

        // 画像を表示するImageViewにURIをセットする
        ImageView imageView = findViewById(R.id.image_view);
        imageView.setImageURI(imageUri);
    }

}
