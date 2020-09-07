#import "Basic";
#import "Compiler";

module_search_paths := string.[
    ".",
];

#run build();

build :: () {
    set_working_directory(#filepath);

    opts := get_build_options();
    opts.output_type = .NO_OUTPUT;
    set_build_options(opts);
    w := compiler_create_workspace("build");
    {
        compiler_begin_intercept(w);
        defer compiler_end_intercept(w);

        array_add(*opts.modules_search_path_array, ..module_search_paths);
        opts.output_type = .EXECUTABLE;
        set_build_options(opts, w);
        add_build_file("test.jai", w);
        while true {
            message := compiler_wait_for_message();
            if !message || message.workspace != w continue;
            if message.kind == .COMPLETE break;
        }
    }

    print("built.\n");
}

