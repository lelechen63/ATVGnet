import os

def main():
    audio_list = os.listdir('/u/lchen63/ATVGnet_cvpr2020/dataset/test_data/test_audio')  
    
    image_list  = os.listdir('/u/lchen63/ATVGnet_cvpr2020/dataset/test_data/test_image')
    
    for imge in image_list:
        if 'DS_Store' in imge:
            continue
        image_path = os.path.join('/u/lchen63/ATVGnet_cvpr2020/dataset/test_data/test_image', imge, '1.jpg')
        
        for audio in audio_list:
            if 'DS_Store' in audio:
                continue
            print (audio)
            audio_path = os.path.join('/u/lchen63/ATVGnet_cvpr2020/dataset/test_data/test_audio', audio)
            
            save_name = audio[:-4] + '__' +  imge + '__1.jpg'
            command = 'python demo.py -i ' + audio_path + ' -p ' + image_path + '  -s ' + save_name
            print  (command)
            os.system(command)
            #python demo.py  -i ../dataset/test_data/test_audio/angry_yousefu_00034.wav -p ../dataset/test_data/test_image/yousef/1.jpg -s angry_yousefu_00034__1

main()
