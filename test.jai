ImGui :: #import "imgui_new";
#import "Basic";

main :: () {
    ctx := ImGui.CreateContext();
    defer ImGui.DestroyContext(ctx);

    io := ImGui.GetIO();

    print("imgui done!\n");
}

