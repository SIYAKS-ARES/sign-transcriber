#!/usr/bin/env python3
"""
Class Mapping Utilities
-----------------------
Merkezi class ID mapping fonksiyonları.
ClassId (1,2,5) <-> Label Index (0,1,2) dönüşümleri.
"""

import numpy as np


def get_class_mapping(target_class_ids):
    """
    Class ID'den index'e mapping oluşturur
    
    Args:
        target_class_ids (list): [1, 2, 5] gibi orijinal class ID'ler
        
    Returns:
        dict: {class_id: index} mapping
        
    Example:
        >>> get_class_mapping([1, 2, 5])
        {1: 0, 2: 1, 5: 2}
    """
    return {cid: idx for idx, cid in enumerate(target_class_ids)}


def get_reverse_mapping(target_class_ids):
    """
    Index'ten class ID'ye reverse mapping oluşturur
    
    Args:
        target_class_ids (list): [1, 2, 5] gibi orijinal class ID'ler
        
    Returns:
        dict: {index: class_id} mapping
        
    Example:
        >>> get_reverse_mapping([1, 2, 5])
        {0: 1, 1: 2, 2: 5}
    """
    return {idx: cid for idx, cid in enumerate(target_class_ids)}


def remap_labels(labels, target_class_ids, to_index=True):
    """
    Label'ları class_id <-> index arasında dönüştürür
    
    Args:
        labels (array-like): Dönüştürülecek label'lar
        target_class_ids (list): Orijinal class ID listesi
        to_index (bool): True ise class_id->index, False ise index->class_id
        
    Returns:
        np.array: Dönüştürülmüş label'lar
        
    Example:
        >>> remap_labels([1, 2, 5, 1], [1, 2, 5], to_index=True)
        array([0, 1, 2, 0])
        
        >>> remap_labels([0, 1, 2, 0], [1, 2, 5], to_index=False)
        array([1, 2, 5, 1])
    """
    labels = np.array(labels)
    
    if to_index:
        # class_id -> index
        mapping = get_class_mapping(target_class_ids)
    else:
        # index -> class_id
        mapping = get_reverse_mapping(target_class_ids)
    
    # Vectorized mapping
    remapped = np.array([mapping[label] for label in labels])
    
    return remapped


def get_original_class_id(index, target_class_ids):
    """
    Index'ten orijinal class ID'yi döndürür
    
    Args:
        index (int): Label index (0-based)
        target_class_ids (list): Orijinal class ID listesi
        
    Returns:
        int: Orijinal class ID
        
    Example:
        >>> get_original_class_id(0, [1, 2, 5])
        1
    """
    return target_class_ids[index]


def validate_class_mapping(labels, target_class_ids, expected_num_classes):
    """
    Class mapping'in doğru olup olmadığını kontrol eder
    
    Args:
        labels (array-like): Kontrol edilecek label'lar
        target_class_ids (list): Hedef class ID'ler
        expected_num_classes (int): Beklenen sınıf sayısı
        
    Returns:
        bool: Geçerliyse True
        
    Raises:
        ValueError: Mapping hatalıysa
    """
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    
    # Label'ların 0'dan başladığını kontrol et
    if unique_labels.min() != 0:
        raise ValueError(
            f"❌ ERROR: Labels should start from 0, but found min={unique_labels.min()}\n"
            f"   Hint: Use remap_labels() to convert class_ids to indices"
        )
    
    # Label'ların ardışık olduğunu kontrol et
    if unique_labels.max() != len(unique_labels) - 1:
        raise ValueError(
            f"❌ ERROR: Labels should be consecutive [0, 1, 2, ...], "
            f"but found: {unique_labels}\n"
            f"   Hint: Check class mapping in data preprocessing"
        )
    
    # Sınıf sayısını kontrol et
    if len(unique_labels) != expected_num_classes:
        raise ValueError(
            f"❌ ERROR: Expected {expected_num_classes} classes, "
            f"but found {len(unique_labels)}\n"
            f"   Unique labels: {unique_labels}\n"
            f"   Target class IDs: {target_class_ids}"
        )
    
    print(f"✅ Class mapping validation passed:")
    print(f"   Labels: {unique_labels.tolist()} (0-indexed)")
    print(f"   Mapped to ClassIDs: {target_class_ids}")
    
    return True


def print_class_distribution(labels, target_class_ids, class_names, split_name=""):
    """
    Sınıf dağılımını yazdırır (güzel formatlanmış)
    
    Args:
        labels (array-like): Label'lar (0-indexed)
        target_class_ids (list): Orijinal class ID'ler
        class_names (list): Sınıf isimleri
        split_name (str): Split adı (Train/Val/Test)
    """
    labels = np.array(labels)
    unique, counts = np.unique(labels, return_counts=True)
    
    if split_name:
        print(f"\n   {split_name}:")
    
    for idx, count in zip(unique, counts):
        original_class_id = target_class_ids[idx]
        class_name = class_names[idx]
        percentage = (count / len(labels) * 100)
        print(f"      Label {idx} (ClassId {original_class_id}, {class_name:10s}): "
              f"{count:3d} ({percentage:5.1f}%)")

