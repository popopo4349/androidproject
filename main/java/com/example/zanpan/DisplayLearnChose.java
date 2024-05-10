package com.example.zanpan;
import android.content.Intent;
import android.os.Bundle;
import android.widget.ImageButton;
import android.widget.Button;
import androidx.activity.EdgeToEdge;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContract;
import android.content.ContentValues;
import android.net.Uri;
import android.provider.MediaStore;
import java.util.Date;
import java.util.concurrent.atomic.AtomicReference;

public class DisplayLearnChose extends AppCompatActivity{
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.learn_display);
        //ホーム画面へ戻る
        ImageButton back_btn = findViewById(R.id.back_btn);
        back_btn.setOnClickListener(v -> finish());


        };


    }



