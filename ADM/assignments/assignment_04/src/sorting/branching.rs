pub fn sort(list: &mut [u32], low: usize, high: usize) {
    if low >= high {
        return;
    }

    let pivot = partition(list, low, high);
    sort(list, low, pivot - 1);
    sort(list, pivot + 1, high);
}

fn partition(list: &mut [u32], low: usize, high: usize) -> usize {
    let pivot = list[high];
    let mut i = low;

    for j in low..high {
        if list[j] <= pivot {
            list.swap(i, j);
            i += 1;
        }
    }

    list.swap(i, high);
    i
}
