package com.example.zanpan;



import androidx.appcompat.app.AppCompatActivity;

import android.net.Uri;
import android.os.Bundle;
import android.widget.ImageView;

public class LearnActivity extends BaseActivity {
    final String PAGE_NAME = "学習モード";
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_learn);
        setupHeader(PAGE_NAME);

        // 画像のURIを受け取る
        String imageUriString = getIntent().getStringExtra("imageUri");
        Uri imageUri = Uri.parse(imageUriString);

        // 画像を表示するImageViewにURIをセットする
        ImageView imageView = findViewById(R.id.image_view);
        imageView.setImageURI(imageUri);
    }

}
