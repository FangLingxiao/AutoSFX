from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip, concatenate_audioclips
import numpy as np

def adjust_volume(clip, volume_factor):
    return clip.volumex(volume_factor)

def mix(clip, object_info):
    if not object_info:
        return clip
    
    area = object_info[0]['area']
    position = object_info[0]['position']

    # Adjust volume based on area
    volume_factor = np.clip(area * 2, 0.5, 2.0)  # Scale area to volume factor
    clip = adjust_volume(clip, volume_factor)

    # Adjust pan based on position
    pan_value = (position[0] - 0.5) * 2  # Map x-position to -1 to 1 range
    clip = clip.audio_pan(pan_value)

    return clip

def merge_audio_video(video_path, effect_infos, ambience_audio, output_path):
    """
    Merge video with multiple audios and output to the specified path.

    parameters:
    video_path (str): path to video folder
    effect_infos (list): list of effect info:
        [
            {
                'interval': (start_frame, end_frame, duration),
                'effect_file': 'path_to_effect_file',
                'duration': duration
                output_path: str
            },
            ...
        ]
    ambience_audio: audio files of ambience
    output_path (str): output folder
    """
    try:
        video_clip = VideoFileClip(video_path)

        if ambience_audio:
            ambience_clip = AudioFileClip(ambience_audio['ambience_file'])
            ambience_clip = ambience_clip.subclip(0, video_clip.duration)
            volume_factor = 10**(-10/20)
            ambience_clip = adjust_volume(ambience_clip, volume_factor)
        else:
            ambience_clip = None
        
        effect_clips = []
        for effect_info in effect_infos:
            effect_clip = AudioFileClip(effect_info['effect_file'])
            
            if effect_info.get('needs_fine_sync', False) and 'fine_sync_starts' in effect_info:
                for i, start_time in enumerate(effect_info['fine_sync_starts']):
                    clip_start = effect_info['interval'][0] / video_clip.fps + start_time
                    if i < len(effect_info['audio_events']):
                        audio_event_time = effect_info['audio_events'][i]
                        audio_end_time = min(audio_event_time + 1, effect_clip.duration)  # 限制音频长度为1秒或直到结束
                        sub_effect_clip = effect_clip.subclip(audio_event_time, audio_end_time)
                        sub_effect_clip = mix(sub_effect_clip, effect_info['object_info'])
                        effect_clips.append(sub_effect_clip.set_start(clip_start))
            else:
                start_time = effect_info['interval'][0] / video_clip.fps
                end_time = min(start_time + effect_clip.duration, video_clip.duration)
                effect_clip = effect_clip.subclip(0, end_time - start_time)
                effect_clip = mix(effect_clip, effect_info['object_info'])
                effect_clips.append(effect_clip.set_start(start_time))

        all_audio_clips = effect_clips
        if ambience_clip:
            all_audio_clips.insert(0, ambience_clip)

        final_audio = CompositeAudioClip(all_audio_clips)
        
        video_clip = video_clip.set_audio(final_audio)
        video_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
        
        return True
    except Exception as e:
        print(f"Error merging audio and video: {e}")
        return False