package com.combatassistant.backend.video;

public class Video {

    private Long id;

    private String filename;

    private String filePath;

    private VideoStatus status;


    public Long getId() {
        return id;
    }


    public void setId(Long id) {
        this.id = id;
    }


    public String getFilename() {
        return filename;
    }


    public void setFilename(String filename) {
        this.filename = filename;
    }


    public String getFilePath() {
        return filePath;
    }


    public void setFilePath(String filePath) {
        this.filePath = filePath;
    }


    public VideoStatus getStatus() {
        return status;
    }


    public void setStatus(VideoStatus status) {
        this.status = status;
    }

}