for f in tests/mojo/*.mojo;
    do echo "=== $f ===";
    pixi run mojo run -I . "$f";
    echo "exit: $?";
done
