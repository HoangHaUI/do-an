Activate env: .\envdoan\Scripts\activate
Check GPU used : nvidia-smi

Progress:
- get imge of bottle from origin image
- classify crop images

convert heic : for file in ./*.HEIC; do heif-convert $file ${file%.HEIC}.jpg; done

Run training.py : 
cd E:\do-an
python src\training\training.py