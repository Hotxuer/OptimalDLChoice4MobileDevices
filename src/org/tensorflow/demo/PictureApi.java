package org.tensorflow.demo;

import okhttp3.MultipartBody;
import okhttp3.RequestBody;
import retrofit2.Call;
import retrofit2.http.Part;
import retrofit2.http.Multipart;
import retrofit2.http.POST;

public interface PictureApi {
    @Multipart
    @POST("post/")
    Call<String> uploadImage(@Part MultipartBody.Part picRgb);
}
