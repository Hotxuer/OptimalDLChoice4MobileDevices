package org.tensorflow.demo;

// 根据状态status是否为success判断请求是否成功的一类请求
public class BasicResponse<T> {
    private String status;
    private String msg;
    private String QRCodeUrl; // 生成支付二维码时返回的二维码网址
    private T result;

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    public String getMsg() {
        return msg;
    }

    public void setMsg(String msg) {
        this.msg = msg;
    }

    public String getQRCodeUrl() {
        return QRCodeUrl;
    }

    public void setQRCodeUrl(String QRCodeUrl) {
        this.QRCodeUrl = QRCodeUrl;
    }

    public T getResult() {
        return result;
    }

    public void setResult(T result) {
        this.result = result;
    }
}
