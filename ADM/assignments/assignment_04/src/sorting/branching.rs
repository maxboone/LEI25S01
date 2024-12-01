pub fn sort(list: &mut [u32]) {
    if list.is_empty() {
        return;
    }

    let pivot = partition(list);
    sort(&mut list[..pivot]);
    sort(&mut list[pivot + 1..]);
}

fn partition(list: &mut [u32]) -> usize {
    let high = list.len() - 1;
    let mut i = 0;

    for j in 0..high {
        if list[j] <= list[high] {
            list.swap(i, j);
            i += 1;
        }
    }

    list.swap(i, high);
    i
}
