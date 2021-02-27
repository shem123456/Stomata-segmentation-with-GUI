# Stomata-segmentation-with-GUI
A GUI for wheat stomata semantic segmentaion
## Download files
You can download the "EXE" file from Baidu clould disk.<p>
Go to the link: https://pan.baidu.com/s/1VRcKm_TEFKkhTJJgksfGdA<p>
Extraction code: abcd<p>
<p>
If this link doesn't work, you can send email: 2020201018@stu.njau.edu.cn.<p>


## Use the Graphical User Interface (GUI)
2.1 open “stomata.exe”<p>

Click on “stomata.exe”, and you will see the GUI.<p>

2.2 Select the video file<p>

Click on the icon  at the top left corner and choose the video you want to test.<p>

2.3 Set up the parameter<p>

Stomata Number: the range is 2 to 10, and it represents the number of stoma you want to choose. Because the first box is used to select the time and date area, its minimun value is 2.<p>
Interval: it represents semantic segmentation every n frames.<p>
Model Path: select a semantic segmentation model. Here, we provide a model (model.h5) for testing.<p>

2.4 Enter the path and name of results<p>

Modify the path and name of the output, and the default are “result.avi” and “result.csv”.<p>

2.5 Select the stomata of interest<p>
Click “Run” , and you can see the first frame of the video.<p>


The first box must select the time and date area, and the other boxes choose the stomata of interest. After drawing a rectangle, press “Enter” on the keyboard to confirm.<p> 

When finished, the “.csv” file and the video file will be saved in the local folder.<p>
