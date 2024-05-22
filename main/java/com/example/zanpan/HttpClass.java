package com.example.zanpan;
import android.os.AsyncTask;
import android.util.Log;
import java.io.File;
import java.io.IOException;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;
public class HttpClass extends AsyncTask<String, Void, String>{   String responseBody;
    public HttpClass(){}

    @Override
    protected String doInBackground(String... params) {
        String url = "送るURL";
        MediaType media = MediaType.parse("multipart/form-data");
        try {
            File file = new File(params[0]);
            String FileName = file.getName();
            String boundary = String.valueOf(System.currentTimeMillis());

            RequestBody requestBody = new MultipartBody.Builder(boundary).setType(MultipartBody.FORM)
                    .addFormDataPart("file", FileName, RequestBody.create(media, file))
                    .build();

            Request request = new Request.Builder()
                    .url(url)
                    .post(requestBody)
                    .build();

            OkHttpClient client = new OkHttpClient();
            Response response = client.newCall(request).execute();
            responseBody = response.body().string();

            return responseBody;

        } catch (IOException e) {
            e.printStackTrace();
        }
        return responseBody;
    }

    @Override
    protected void onPostExecute(String result) {
        Log.d("a",result);
    }
}

