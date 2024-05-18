package com.example.zanpan;
import android.content.ContentValues;
import android.content.DialogInterface;
import android.content.Intent;
import android.net.Uri;
import android.provider.MediaStore;
import android.widget.ImageButton;
import android.widget.TextView;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import android.view.View;

import android.net.Uri;

public class BaseActivity extends AppCompatActivity{

    protected void setupHeader() {
        ImageButton home_btn = findViewById(R.id.home_btn);
        TextView pageNameTextView = findViewById(R.id.page_name);

        home_btn.setVisibility(View.GONE);
        pageNameTextView.setVisibility(View.GONE);

    }
    protected void setupHeader(String pagename) {
        ImageButton home_btn = findViewById(R.id.home_btn);
        home_btn.setOnClickListener(v -> {
            Intent intent = new Intent(BaseActivity.this, MainActivity.class);
            startActivity(intent);
        });
        TextView pageNameTextView = findViewById(R.id.page_name);
        pageNameTextView.setText(pagename);
    }



}
