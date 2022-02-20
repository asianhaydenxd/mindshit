if true
    print('a')
end

if false
    print('a')
elif true
    print('b')
else
    print('c')
end

if false
    print('a')
else
    print('b')
end

if true
    if false
        if false
            print('a')
        else
            print('b')
        end
    elif true
        print('c')
    else
        print('d')
    end
else
    print('e')
end