from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_audioclips

def merge_audio_video(video_path, audio_infos, output_path):
    """
    合并视频和多个音频，并输出到指定路径。

    参数:
    video_path (str): 视频文件路径。
    audio_infos (list): 包含音频文件信息的列表，格式如下:
        [
            {
                'interval': (start_frame, end_frame, duration),
                'audio_file': 'path_to_audio_file',
                'duration': duration
            },
            ...
        ]
    output_path (str): 输出合并后的视频文件路径。
    """
    try:
        # 加载视频
        video_clip = VideoFileClip(video_path)
        
        # 加载并剪辑音频
        audio_clips = []
        for audio_info in audio_infos:
            audio_clip = AudioFileClip(audio_info['audio_file'])
            audio_clip = audio_clip.subclip(0, min(audio_clip.duration, video_clip.duration))
            audio_clips.append(audio_clip)
        
        # 合并音频片段
        final_audio_clip = concatenate_audioclips(audio_clips)
        
        # 将音频设置为视频的音频
        video_clip = video_clip.set_audio(final_audio_clip)
        
        # 导出合并后的文件
        video_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
        
        return True
    except Exception as e:
        print(f"Error merging audio and video: {e}")
        return False