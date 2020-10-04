#include <algorithm>

#define NDEBUG
#include <assert.h>
#include <iostream>

template<typename T> class GroupedArray;
template<typename T> std::ostream& operator<<(std::ostream& os, const GroupedArray<T>& array);

template <typename T>
class GroupedArray {
public:
	void add(T value, int groupIndex);
	void remove(int index, int groupIndex);
	void transfer(int index, int oldGroup, int newGroup);

	void transferForward(int index, int oldGroup, int newGroup);
	void transferBackward(int index, int oldGroup, int newGroup);

	int groupStart(int groupIndex) const;
	int groupSize(int groupIndex) const;
	int* groupSize(); // don't modify this directly, it's here as non-const for SIMD purposes
	int groupsCount() const;

	T& operator[](int i);
	const T& operator[](int i) const;

	friend std::ostream& operator<< <>(std::ostream& os, const GroupedArray<T>& array);

	GroupedArray(int size, int groups);
	~GroupedArray();

	T* data;

private:
	int size;
	int groups;

	int* groupIndecies;
	int* groupSizes;

	void _add(T value, int groupIndex, int groupStop);
	void _remove(int index, int groupIndex, int groupStop);

};

template <typename T>
GroupedArray<T>::GroupedArray(int size, int groups) {
	this->size = size;
	this->groups = groups;

	data = new T[size];
	groupIndecies = new int[groups+1];
	groupSizes = new int[groups];
	std::fill(groupIndecies, groupIndecies + groups + 1, 0);
	std::fill(groupSizes, groupSizes + groups, 0);
}

template <typename T>
GroupedArray<T>::~GroupedArray() {
	delete[] data;
	delete[] groupIndecies;
	delete[] groupSizes;
}

template <typename T>
void GroupedArray<T>::add(T value, int groupIndex) {
	assert(groupIndex >= 0 && groupIndex < groups);
	_add(value, groupIndex, groups);
}

template <typename T>
void GroupedArray<T>::remove(int index, int groupIndex) {
	assert(index >= 0 && index < size);
	assert(groupIndex >= 0 && groupIndex < groups);
	assert(index >= groupIndecies[groupIndex] && index < groupIndecies[groupIndex+1]);
	_remove(index, groupIndex, groups);
}

template <typename T>
void GroupedArray<T>::_add(T value, int groupIndex, int groupStop) {
	for (int i = groupStop-1; i >= groupIndex; i--) {
		if (groupSizes[i] > 0) { // ignore group without any values
			data[groupIndecies[i+1]] = data[groupIndecies[i]]; // copy first value in this group forward
		}
	}
	data[groupIndecies[groupIndex]] = value; // do actual insertion
	groupSizes[groupIndex]++;

	for (int i = groupIndex+1; i < groupStop+1; i++) {
		groupIndecies[i]++; // shift all following groups forward
	}
}

template <typename T>
void GroupedArray<T>::_remove(int index, int groupIndex, int groupStop) {
	data[index] = data[groupIndecies[groupIndex+1]-1]; // copy last item over item to be removed
	groupSizes[groupIndex]--;

	for (int i = groupIndex+1; i < groupStop; i++) {
		if (groupSizes[i] > 0) { // only copy into a group if it exists
			data[groupIndecies[i]-1] = data[groupIndecies[i+1]-1]; // put this group's last val before its first
		}
	}

	for (int i = groupIndex+1; i < groupStop+1; i++) {
		groupIndecies[i]--; // shift all following groups back
	}
}

template <typename T>
void GroupedArray<T>::transfer(int index, int oldGroup, int newGroup) {
	assert(index >= 0 && index < size);
	assert(oldGroup >= 0 && oldGroup < groups);
	assert(newGroup >= 0 && newGroup < groups);
	assert(newGroup != oldGroup);
	assert(index >= groupIndecies[oldGroup] && index < groupIndecies[oldGroup+1]);

	if (oldGroup < newGroup) {
		transferForward(index, oldGroup, newGroup);
	} else {
		transferBackward(index, oldGroup, newGroup);
	}
}

template <typename T>
void GroupedArray<T>::transferForward(int index, int oldGroup, int newGroup) {
	assert(index >= 0 && index < size);
	assert(oldGroup >= 0 && oldGroup < groups);
	assert(newGroup >= 0 && newGroup < groups);
	assert(newGroup != oldGroup);
	assert(index >= groupIndecies[oldGroup] && index < groupIndecies[oldGroup+1]);
	assert(oldGroup < newGroup);

	T value = data[index];
	_remove(index, oldGroup, newGroup);
	groupSizes[newGroup]++;
	data[groupIndecies[newGroup]] = value;
}

template <typename T>
void GroupedArray<T>::transferBackward(int index, int oldGroup, int newGroup) {
	assert(index >= 0 && index < size);
	assert(oldGroup >= 0 && oldGroup < groups);
	assert(newGroup >= 0 && newGroup < groups);
	assert(newGroup != oldGroup);
	assert(index >= groupIndecies[oldGroup] && index < groupIndecies[oldGroup+1]);
	assert(oldGroup > newGroup);

	T value = data[index];
	std::swap(data[index], data[groupIndecies[oldGroup]]);
	_add(value, newGroup, oldGroup);
	groupSizes[oldGroup]--;
}

template <typename T>
T& GroupedArray<T>::operator[](int i) {
	assert(i >= 0 && i < size);
	return data[i];
}

template <typename T>
const T& GroupedArray<T>::operator[](int i) const {
	assert(i >= 0 && i < size);
	return data[i];
}

template <typename T>
int GroupedArray<T>::groupStart(int groupIndex) const {
	assert(groupIndex >= 0 && groupIndex <= groups);
	return groupIndecies[groupIndex];
}

template <typename T>
int GroupedArray<T>::groupSize(int groupIndex) const {
	assert(groupIndex >= 0 && groupIndex < groups);
	return groupSizes[groupIndex];
}

template <typename T>
int* GroupedArray<T>::groupSize() {
	return groupSizes;
}

template <typename T>
int GroupedArray<T>::groupsCount() const {
	return groups;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const GroupedArray<T>& array) {
	for (int i = 0; i < array.groupsCount(); i++) {
		os << " |";
		if (array.groupSize(i) > 0) {
			os << array[array.groupStart(i)];
		}
		for (int j = 1; j < array.groupSize(i); j++) {
			os << ", " << array[j + array.groupStart(i)];
		}
	}
	return os;
}