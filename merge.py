from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip, concatenate_audioclips
import numpy as np

def adjust_volume(clip, volume_factor):
    return clip.volumex(volume_factor)

def merge_audio_video(video_path, effect_infos, ambience_audio, output_path):
    """
    Merge video with multiple audios and output to the specified path.

    参数:
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
        # load video
        video_clip = VideoFileClip(video_path)
        
        # load and edit audio
        effect_clips = []
        for effect_info in effect_infos:
            effect_clip = AudioFileClip(effect_info['effect_file'])
            effect_clip = effect_clip.subclip(0, min(effect_clip.duration, video_clip.duration))
            effect_clips.append(effect_clip)

        # Merge sound effects
        if effect_clips:
            effect_audio = concatenate_audioclips(effect_clips)
        else:
            effect_audio = None
        
        # Process effect audio
        if ambience_audio:
            ambience_clip =  AudioFileClip(ambience_audio['ambience_file'])
            # edit ambience duration
            ambience_clip = ambience_clip.subclip(0, video_clip.duration)
            # edit ambience volume
            volume_factor = 10**(-10/20) # convert dB to linear scale factor
            ambience_clip = adjust_volume(ambience_clip, volume_factor)

            # Merge effect and ambience
            if effect_audio:
                final_audio = CompositeAudioClip([effect_audio, ambience_clip])
            else:
                final_audio = ambience_clip
            
        else:
            final_audio = effect_audio if effect_audio else None
        
        # Set final audio as audio for video
        if final_audio:
            video_clip = video_clip.set_audio(final_audio)

        # Output        
        video_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
        
        return True
    except Exception as e:
        print(f"Error merging audio and video: {e}")
        return False