cl /nologo /c /Zi /Os dllmain.cpp imgui.cpp imgui_demo.cpp imgui_draw.cpp imgui_widgets.cpp /I. && ^
link /nologo *.obj /out:imgui.dll /incremental:no /debug /dll /machine:amd64
