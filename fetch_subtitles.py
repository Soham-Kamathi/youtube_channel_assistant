import shutil
import os
from datetime import datetime
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs

def get_video_id(url):
    """Extract video ID from YouTube URL."""
    parsed_url = urlparse(url)
    if parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
        if parsed_url.path == '/watch':
            return parse_qs(parsed_url.query)['v'][0]
    elif parsed_url.hostname == 'youtu.be':
        return parsed_url.path[1:]
    return None

def process_urls(urls_file):
    """Process URLs from the given file and download their subtitles."""
    successful = 0
    failed = 0
    
    # Create output directory if it doesn't exist
    os.makedirs('youtube_subtitles', exist_ok=True)
    
    try:
        with open(urls_file, 'r') as file:
            urls = [line.strip() for line in file if line.strip()]
            
        if not urls:
            print("No URLs found in the file.")
            return successful, failed
            
        for url in urls:
            try:
                video_id = get_video_id(url)
                if not video_id:
                    raise ValueError("Invalid YouTube URL")
                
                # Get the transcript
                transcript = YouTubeTranscriptApi.get_transcript(video_id)
                
                # Save to file
                output_file = os.path.join('youtube_subtitles', f'{video_id}.txt')
                with open(output_file, 'w', encoding='utf-8') as f:
                    # Add the video URL as the first line
                    f.write(f"Video URL: {url}\n\n")
                    # Write the subtitles
                    for entry in transcript:
                        f.write(f"{entry['text']}\n")
                
                successful += 1
                print(f"Successfully downloaded subtitles for: {url}")
                
            except Exception as e:
                failed += 1
                with open('failed_videos.txt', 'a', encoding='utf-8') as f:
                    f.write(f"{url}: {str(e)}\n")
                print(f"Failed to process: {url}")
                
        return successful, failed
        
    except FileNotFoundError:
        print(f"Error: File '{urls_file}' not found.")
        return successful, failed

def create_subtitle_zip():
    """Create a zip file of all subtitle files in the youtube_subtitles directory."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    zip_filename = f'youtube_subtitles_{timestamp}.zip'

    try:
        shutil.make_archive(
            base_name=f'youtube_subtitles_{timestamp}',
            format='zip',
            root_dir='youtube_subtitles'
        )
        print(f"\nZip file created successfully: {zip_filename}")
        return zip_filename
    except Exception as e:
        print(f"\nError creating zip file: {str(e)}")
        return None

def main():
    """Main function to run the subtitle fetcher."""
    urls_file = 'youtube_urls.txt'
    print("Starting YouTube subtitle fetcher...")

    successful, failed = process_urls(urls_file)

    print("\nSummary:")
    print(f"Successfully processed: {successful} videos")
    print(f"Failed: {failed} videos")
    print("\nCheck 'failed_videos.txt' for details on failed attempts")
    print(f"Subtitles have been saved in the 'youtube_subtitles' directory")

    zip_file = create_subtitle_zip()
    if zip_file:
        print(f"All subtitles have been zipped to: {zip_file}")

if __name__ == "__main__":
    main()