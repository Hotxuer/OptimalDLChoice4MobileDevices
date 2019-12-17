package org.tensorflow.demo;

import java.net.Socket;

import retrofit2.Retrofit;

public class Store {
    private static boolean whetherLocal = true;
    private static boolean whetherAuto = false;
    private static Retrofit retrofit;
//    public static final String serverUrl = "http://172.18.2.18:8000/";
    public static final String serverUrl = "http://172.18.2.13:8000/";

    public static final String ip = "172.18.9.62";
    public static final int port = 6666;

    public static boolean isDrawing = false;

    public static Socket socket;

    public static boolean isComputing = false;

    private static long networkDelay = 0;

    private static Integer sendInterval = 10;

    private static Integer imageQuality = 100;

    public static Integer getSendInterval() {
        return sendInterval;
    }

    public static void setSendInterval(Integer sendInterval) {
        Store.sendInterval = sendInterval;
    }

    public static Integer getImageQuality() {
        return imageQuality;
    }

    public static void setImageQuality(Integer imageQuality) {
        Store.imageQuality = imageQuality;
    }

    public static long getNetworkDelay() {
        return networkDelay;
    }

    public static void setNetworkDelay(long networkDelay) {
        Store.networkDelay = networkDelay;
    }

    public static boolean isWhetherAuto() {
        return whetherAuto;
    }

    public static void setWhetherAuto(boolean whetherAuto) {
        Store.whetherAuto = whetherAuto;
    }

    public static Socket getSocket() {
        return socket;
    }

    public static void setSocket(Socket socket) {
        Store.socket = socket;
    }

    public static boolean isIsComputing() {
        return isComputing;
    }

    public static void setIsComputing(boolean isComputing) {
        Store.isComputing = isComputing;
    }

    public static boolean isWhetherLocal() {
        return whetherLocal;
    }

    public static void setWhetherLocal(boolean local) {
        Store.whetherLocal = local;
    }

    public static Retrofit getRetrofit() {
        return retrofit;
    }

    public static void setRetrofit(Retrofit retrofit) {
        Store.retrofit = retrofit;
    }
}
