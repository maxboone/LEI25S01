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
        let c = list[j] <= pivot;
        let t = list[i];
        list[i] = list[j] * c as u32 + t * (1 - c as u32);
        list[j] = t * c as u32 + list[j] * (1 - c as u32);
        i += c as usize;
    }

    list.swap(i, high);
    i
}
