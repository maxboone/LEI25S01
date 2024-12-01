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
        let c = list[j] <= list[high];
        let t = list[i];
        list[i] = list[j] * c as u32 + t * (1 - c as u32);
        list[j] = t * c as u32 + list[j] * (1 - c as u32);
        i += c as usize;
    }

    list.swap(i, high);
    i
}
