CC = cl
CFLAGS = /MD /EHsc
INCLUDES = /I "include" /I "..\onnxruntime-win-x64-1.20.1\include" /I "..\opencv\build\include"
LIBPATH = /link /LIBPATH:"..\opencv\build\x64\vc16\lib" /LIBPATH:"..\onnxruntime-win-x64-1.20.1\lib"
LIBS = opencv_world4110.lib onnxruntime.lib
SRCDIR = src

imgDisplay:
	$(CC) $(CFLAGS) $(INCLUDES) $(SRCDIR)/imgDisplay.cpp $(SRCDIR)/filter.cpp /Fobin/ /Febin/$@ $(LIBPATH) $(LIBS)

vidDisplay:
	$(CC) $(CFLAGS) $(INCLUDES) $(SRCDIR)/vidDisplay.cpp $(SRCDIR)/filter.cpp $(SRCDIR)/faceDetect.cpp /Fobin/ /Febin/$@ $(LIBPATH) $(LIBS)

timeBlur:
	$(CC) $(CFLAGS) $(INCLUDES) $(SRCDIR)/timeBlur.cpp $(SRCDIR)/filter.cpp /Fobin/ /Febin/$@ $(LIBPATH) $(LIBS)

da2vid:
	$(CC) $(CFLAGS) $(INCLUDES) $(SRCDIR)/da2-video.cpp /Fobin/ /Febin/$@ $(LIBPATH) $(LIBS)

runda2vid: da2vid
	.\bin\da2vid.exe

runImg: imgDisplay
	.\bin\imgDisplay.exe data\melon.jpg

runVid: vidDisplay
	.\bin\vidDisplay.exe

runTimeBlur: timeBlur
	.\bin\timeBlur.exe data/cathedral.jpeg

clean:
	del bin\*.obj bin\*.exe *.jpg