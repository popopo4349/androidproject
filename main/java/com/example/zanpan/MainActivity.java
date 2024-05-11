package com.example.zanpan;
import android.graphics.Bitmap;
import android.content.ContentValues;
import android.content.Intent;
import android.os.Bundle;
import android.provider.MediaStore;
import android.widget.Button;
import androidx.activity.EdgeToEdge;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import java.io.IOException;
import android.net.Uri;
import androidx.appcompat.app.AlertDialog;
import android.content.DialogInterface;


public class MainActivity extends AppCompatActivity {
    static final int REQUEST_IMAGE_CAPTURE = 1;
    private Class<?> activityClass;
    private static final int PICK_IMAGE_REQUEST = 1;
    Uri imageUri;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);

        Button learn_btn = findViewById(R.id.learn_btn);
        learn_btn.setOnClickListener(v -> {
            activityClass = LearnActivity.class;
            showOptionDialog();
        });

        Button infer_btn = findViewById(R.id.infer_btn);
        infer_btn.setOnClickListener(v -> {
            activityClass = InferActivity.class;
            showOptionDialog();
        });

    }

    ActivityResultLauncher<Intent> startForResultTake = registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(),
            result -> {
                if (result.getResultCode() == RESULT_OK) {
                    Intent intent = new Intent(this, activityClass);
                    intent.putExtra("imageUri", imageUri.toString());
                    startActivity(intent);
                }
            });

    ActivityResultLauncher<Intent> startForResultLoad = registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(),
            result -> {
                    Intent data = result.getData();
                    if (data != null && data.getData() != null) {
                        Uri uri = data.getData();
                        Intent intent = new Intent(this, activityClass);
                        intent.putExtra("imageUri", uri.toString());
                        startActivity(intent);
                    }
                }
    );

    private void showOptionDialog() {
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        final CharSequence[] items = {"カメラから画像を撮影する", "フォルダーから画像を選ぶ", "戻る"};
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
                            case 2:
                                dialog.dismiss();
                                break;
                        }
                    }
                });
        builder.create().show();
    }

    private void takePicture() {
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

    private void loadPicture() {
        Intent intent = new Intent(Intent.ACTION_PICK, android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startForResultLoad.launch(Intent.createChooser(intent, "Select Picture"));
    }

}