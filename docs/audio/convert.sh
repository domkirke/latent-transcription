FILES_TO_CONVERT=`find . -name *.wav`

for file in ${FILES_TO_CONVERT[@]}
do
	ffmpeg -i ${file} ${file%.*}.mp3
done
