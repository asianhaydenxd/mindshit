a: &0 = 10
num: &1

while &0
    a -= 1
    num = a
    num += 48
    out num
    &5 = 5
    while &5
        &5 -= 1
        &6 = &5
        &6 += 48
        out &6
    end
end