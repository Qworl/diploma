on run argv
    set docPath to POSIX file (item 1 of argv) as alias
    set pdfPosix to item 2 of argv
    tell application "Microsoft Word"
        activate
        open docPath
        set theDoc to active document
        save as theDoc file name pdfPosix file format format PDF
        close theDoc saving no
    end tell
end run
