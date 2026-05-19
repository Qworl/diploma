on run argv
    set docPosix to item 1 of argv
    set docFile to POSIX file docPosix as alias
    tell application "Microsoft Word"
        activate
        open docFile
        set theDoc to active document
        repaginate theDoc
        set introPage to my findHeading(theDoc, "Введение")
        set endPage to my findHeading(theDoc, "Список использованных источников")
        if endPage is 0 then set endPage to my findHeading(theDoc, "Приложение А")
        if endPage is 0 then set endPage to (compute statistics theDoc statistic statistic pages) + 1
        close theDoc saving no
    end tell
    set mainPages to endPage - introPage
    return (introPage as string) & " " & (endPage as string) & " " & (mainPages as string)
end run

-- Ищет первое вхождение `prefix` в параграфе со стилем Heading 1, возвращает номер страницы.
on findHeading(theDoc, prefix)
    tell application "Microsoft Word"
        set startRange to create range theDoc start 0 end 0
        select startRange
        set foundOK to false
        try
            tell selection
                set theFinder to find object
                clear formatting theFinder
                set match case of theFinder to true
                set forward of theFinder to true
                set wrap of theFinder to find stop
                set style of theFinder to style heading1
                set foundOK to (execute find theFinder find text prefix)
            end tell
        on error
            set foundOK to false
        end try
        if not foundOK then return 0
        try
            set r to selection of active window
            return (get range information r information type active end page number) as integer
        on error
            return 0
        end try
    end tell
end findHeading
