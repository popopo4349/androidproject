package com.example.zanpan;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;

import android.content.ContentValues;
import android.net.Uri;
import android.app.Activity;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.content.Intent;
import android.provider.MediaStore;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;
import java.util.Date;
import java.util.concurrent.atomic.AtomicReference;

public class CameraX extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_camera);
        AtomicReference<Uri> uri = new AtomicReference<Uri>();
        ImageView img = findViewById(R.id.image_view);

        ActivityResultLauncher<Intent> startForResult = registerForActivityResult(
                new ActivityResultContracts.StartActivityForResult(), result -> {
                    if(result.getResultCode()==RESULT_OK){
                        img.setImageURI(uri.get());
                    }
                }
        );

        //imageView = findViewById(R.id.image_view);

        Button cameraButton = findViewById(R.id.camera_btn);
        // lambda式
        cameraButton.setOnClickListener( v -> {
            ContentValues cv = new ContentValues();
            cv.put(MediaStore.Images.Media.TITLE, "mypic-" + new Date().getTime() + ".jpg");
            cv.put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg");
            uri.set(getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI,cv));

            Intent i = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
            i.putExtra(MediaStore.EXTRA_OUTPUT, uri.get());
            startForResult.launch(i);
            //resultLauncher.launch(i);
        });
    }
}