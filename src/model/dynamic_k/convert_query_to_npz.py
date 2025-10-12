#!/usr/bin/env python3
"""
Convert binary files to NPZ format for DynamicKClusterDistTransformer queries_path

Supports multiple binary file formats:
1. fvecs format: Standard format for SIFT/GIST feature vectors
2. bvecs format: Byte vector format  
3. raw binary array format: Continuously stored float32 or other data types
4. custom format: Binary files with header information

Output NPZ file contains:
- 'queries': float32 array with shape [N, D], compatible with DynamicKClusterDistTransformer

Usage examples:

# Convert fvecs file
python src/model/dynamic_k/convert_bin_to_npz.py \
    --input_bin input/vectors.fvecs \
    --output_npz output/queries.npz \
    --format fvecs

# Convert raw binary file
python src/model/dynamic_k/convert_bin_to_npz.py \
    --input_bin input/vectors.bin \
    --output_npz output/queries.npz \
    --format raw \
    --dtype float32 \
    --shape 1000 960
"""

import os
import numpy as np
import argparse
import struct
from pathlib import Path
from typing import Tuple, Optional


def read_fvecs(file_path: str, max_vectors: Optional[int] = None) -> np.ndarray:
    print(f"Reading fvecs file: {file_path}")
    
    vectors = []
    with open(file_path, 'rb') as f:
        vector_count = 0
        while True:
            # Read dimension
            dim_bytes = f.read(4)
            if len(dim_bytes) != 4:
                break  # End of file
            
            dim = struct.unpack('i', dim_bytes)[0]
            
            # Read vector data
            vector_bytes = f.read(dim * 4)
            if len(vector_bytes) != dim * 4:
                print(f"Warning: Incomplete vector at position {vector_count}")
                break
            
            vector = struct.unpack(f'{dim}f', vector_bytes)
            vectors.append(vector)
            
            vector_count += 1
            
            # Print progress every 100k vectors
            if vector_count % 100000 == 0:
                print(f"  Read {vector_count} vectors")
            
            # Check maximum vector limit
            if max_vectors is not None and vector_count >= max_vectors:
                print(f"  Reached maximum vector limit: {max_vectors}")
                break
    
    if not vectors:
        raise ValueError(f"No vectors found in {file_path}")
    
    vectors_array = np.array(vectors, dtype=np.float32)
    print(f"  Loaded {len(vectors)} vectors, shape: {vectors_array.shape}")
    
    return vectors_array


def read_bvecs(file_path: str, max_vectors: Optional[int] = None) -> np.ndarray:
    """
    Read bvecs format file (byte vector format)
    
    bvecs format:
    - First 4 bytes of each vector: dimension d (int32)
    - Next d bytes: vector data (uint8)
    
    Args:
        file_path: bvecs file path
        max_vectors: maximum number of vectors to read
    
    Returns:
        vectors: float32 array with shape [N, D]
    """
    print(f"Reading bvecs file: {file_path}")
    
    vectors = []
    with open(file_path, 'rb') as f:
        vector_count = 0
        while True:
            # Read dimension
            dim_bytes = f.read(4)
            if len(dim_bytes) != 4:
                break  # End of file
            
            dim = struct.unpack('i', dim_bytes)[0]
            
            # Read vector data
            vector_bytes = f.read(dim)
            if len(vector_bytes) != dim:
                print(f"Warning: Incomplete vector at position {vector_count}")
                break
            
            vector = struct.unpack(f'{dim}B', vector_bytes)
            vectors.append(vector)
            
            vector_count += 1
            
            # Print progress every 100k vectors
            if vector_count % 100000 == 0:
                print(f"  Read {vector_count} vectors")
            
            # Check maximum vector limit
            if max_vectors is not None and vector_count >= max_vectors:
                print(f"  Reached maximum vector limit: {max_vectors}")
                break
    
    if not vectors:
        raise ValueError(f"No vectors found in {file_path}")
    
    # Convert to float32
    vectors_array = np.array(vectors, dtype=np.float32)
    print(f"  Loaded {len(vectors)} vectors, shape: {vectors_array.shape}")
    
    return vectors_array


def read_raw_binary(
    file_path: str, 
    dtype: str = "float32",
    shape: Optional[Tuple[int, int]] = None,
    max_vectors: Optional[int] = None
) -> np.ndarray:
    """
    Read raw binary file
    """
    print(f"Reading raw binary file: {file_path}")
    print(f"  Data type: {dtype}")
    
    # Load raw data
    data = np.fromfile(file_path, dtype=dtype)
    print(f"  Total elements: {len(data)}")
    
    if shape is not None:
        n_vectors, dim = shape
        if len(data) != n_vectors * dim:
            print(f"Warning: Data size ({len(data)}) doesn't match specified shape ({n_vectors}×{dim}={n_vectors*dim})")
            # Try auto-adjustment
            if len(data) % dim == 0:
                n_vectors = len(data) // dim
                print(f"  Auto-adjusting to {n_vectors} vectors of dimension {dim}")
            else:
                raise ValueError(f"Cannot reshape data of length {len(data)} to shape with dimension {dim}")
        
        # Apply maximum vector limit
        if max_vectors is not None and n_vectors > max_vectors:
            print(f"  Limiting to first {max_vectors} vectors")
            data = data[:max_vectors * dim]
            n_vectors = max_vectors
        
        vectors = data.reshape(n_vectors, dim)
    else:
        # Try auto-inferring shape
        print("  Auto-detecting shape...")
        possible_dims = []
        for dim in [128, 256, 384, 512, 960, 1024, 2048]:  # Common dimensions
            if len(data) % dim == 0:
                possible_dims.append((len(data) // dim, dim))
        
        if not possible_dims:
            raise ValueError(f"Cannot infer shape from data length {len(data)}. Please specify --shape.")
        
        if len(possible_dims) == 1:
            n_vectors, dim = possible_dims[0]
            print(f"  Detected shape: ({n_vectors}, {dim})")
        else:
            print(f"  Multiple possible shapes: {possible_dims}")
            # Choose the most reasonable shape (moderate number of vectors)
            best_shape = min(possible_dims, key=lambda x: abs(x[0] - 100000))  # Prefer around 100k vectors
            n_vectors, dim = best_shape
            print(f"  Using shape: ({n_vectors}, {dim})")
        
        # Apply maximum vector limit
        if max_vectors is not None and n_vectors > max_vectors:
            print(f"  Limiting to first {max_vectors} vectors")
            data = data[:max_vectors * dim]
            n_vectors = max_vectors
        
        vectors = data.reshape(n_vectors, dim)
    
    # Convert to float32
    vectors = vectors.astype(np.float32)
    print(f"  Final shape: {vectors.shape}")
    
    return vectors


def read_custom_binary(
    file_path: str,
    header_size: int = 0,
    dtype: str = "float32",
    shape: Optional[Tuple[int, int]] = None,
    max_vectors: Optional[int] = None
) -> np.ndarray:
    """
    Read custom binary file with header information
    
    Args:
        file_path: binary file path
        header_size: number of header bytes to skip
        dtype: data type
        shape: data shape (N, D)
        max_vectors: maximum number of vectors to read
    
    Returns:
        vectors: float32 array with shape [N, D]
    """
    print(f"Reading custom binary file: {file_path}")
    print(f"  Header size: {header_size} bytes")
    print(f"  Data type: {dtype}")
    
    with open(file_path, 'rb') as f:
        # Skip header
        if header_size > 0:
            f.read(header_size)
        
        # Read remaining data
        remaining_data = f.read()
    
    # Convert to numpy array
    data = np.frombuffer(remaining_data, dtype=dtype)
    print(f"  Data elements: {len(data)}")
    
    # Use same shape processing logic as raw_binary
    if shape is not None:
        n_vectors, dim = shape
        if len(data) != n_vectors * dim:
            print(f"Warning: Data size ({len(data)}) doesn't match specified shape ({n_vectors}×{dim}={n_vectors*dim})")
            if len(data) % dim == 0:
                n_vectors = len(data) // dim
                print(f"  Auto-adjusting to {n_vectors} vectors of dimension {dim}")
            else:
                raise ValueError(f"Cannot reshape data of length {len(data)} to shape with dimension {dim}")
        
        # Apply maximum vector limit
        if max_vectors is not None and n_vectors > max_vectors:
            print(f"  Limiting to first {max_vectors} vectors")
            data = data[:max_vectors * dim]
            n_vectors = max_vectors
        
        vectors = data.reshape(n_vectors, dim)
    else:
        # Auto-infer shape (same logic as raw_binary)
        print("  Auto-detecting shape...")
        possible_dims = []
        for dim in [128, 256, 384, 512, 960, 1024, 2048]:
            if len(data) % dim == 0:
                possible_dims.append((len(data) // dim, dim))
        
        if not possible_dims:
            raise ValueError(f"Cannot infer shape from data length {len(data)}. Please specify --shape.")
        
        if len(possible_dims) == 1:
            n_vectors, dim = possible_dims[0]
            print(f"  Detected shape: ({n_vectors}, {dim})")
        else:
            print(f"  Multiple possible shapes: {possible_dims}")
            best_shape = min(possible_dims, key=lambda x: abs(x[0] - 100000))
            n_vectors, dim = best_shape
            print(f"  Using shape: ({n_vectors}, {dim})")
        
        # Apply maximum vector limit
        if max_vectors is not None and n_vectors > max_vectors:
            print(f"  Limiting to first {max_vectors} vectors")
            data = data[:max_vectors * dim]
            n_vectors = max_vectors
        
        vectors = data.reshape(n_vectors, dim)
    
    # Convert to float32
    vectors = vectors.astype(np.float32)
    print(f"  Final shape: {vectors.shape}")
    
    return vectors


def convert_bin_to_npz(
    input_bin_path: str,
    output_npz_path: str,
    format_type: str = "auto",
    dtype: str = "float32",
    shape: Optional[Tuple[int, int]] = None,
    header_size: int = 0,
    max_vectors: Optional[int] = None,
    array_name: str = "queries",
    verify: bool = True
) -> None:
    """
    Convert binary file to NPZ format
    
    Args:
        input_bin_path: input binary file path
        output_npz_path: output NPZ file path
        format_type: file format ("auto", "fvecs", "bvecs", "raw", "custom")
        dtype: data type (for raw and custom formats)
        shape: data shape (N, D)
        header_size: header bytes (for custom format)
        max_vectors: maximum number of vectors
        array_name: array name in NPZ file
        verify: whether to verify output file
    """
    if not os.path.exists(input_bin_path):
        raise FileNotFoundError(f"Input file not found: {input_bin_path}")
    
    print(f"Converting {input_bin_path} to {output_npz_path}")
    print(f"Format: {format_type}")
    
    # Auto-detect format
    if format_type == "auto":
        file_ext = Path(input_bin_path).suffix.lower()
        if file_ext == ".fvecs":
            format_type = "fvecs"
        elif file_ext == ".bvecs":
            format_type = "bvecs"
        else:
            format_type = "raw"
            print(f"  Auto-detected format: {format_type}")
    
    # Read data based on format
    if format_type == "fvecs":
        vectors = read_fvecs(input_bin_path, max_vectors)
    elif format_type == "bvecs":
        vectors = read_bvecs(input_bin_path, max_vectors)
    elif format_type == "raw":
        vectors = read_raw_binary(input_bin_path, dtype, shape, max_vectors)
    elif format_type == "custom":
        vectors = read_custom_binary(input_bin_path, header_size, dtype, shape, max_vectors)
    else:
        raise ValueError(f"Unsupported format: {format_type}")
    
    # Validate data
    if vectors.size == 0:
        raise ValueError("No data loaded from input file")
    
    print(f"\nData Summary:")
    print(f"  Shape: {vectors.shape}")
    print(f"  Data type: {vectors.dtype}")
    print(f"  Value range: [{vectors.min():.6f}, {vectors.max():.6f}]")
    print(f"  Memory usage: {vectors.nbytes / 1024 / 1024:.2f} MB")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_npz_path), exist_ok=True)
    
    # Save to NPZ format
    print(f"\nSaving to NPZ format: {output_npz_path}")
    save_data = {array_name: vectors}
    np.savez_compressed(output_npz_path, **save_data)
    
    print(f"✅ Successfully converted to NPZ format")
    print(f"   Output file: {output_npz_path}")
    print(f"   Array name: '{array_name}'")
    print(f"   Array shape: {vectors.shape}")
    print(f"   File size: {os.path.getsize(output_npz_path) / 1024 / 1024:.2f} MB")
    
    # Verify output file
    if verify:
        print(f"\n--- Verification ---")
        try:
            loaded = np.load(output_npz_path)
            print(f"NPZ file contains keys: {list(loaded.keys())}")
            
            for key in loaded.keys():
                array = loaded[key]
                print(f"  {key}: {array.shape}, dtype: {array.dtype}")
                if array.size > 0:
                    print(f"    Value range: [{array.min():.6f}, {array.max():.6f}]")
                    print(f"    Sample (first vector): {array[0][:5]}{'...' if array.shape[1] > 5 else ''}")
            
            # Verify data consistency
            loaded_array = loaded[array_name]
            if np.array_equal(vectors, loaded_array):
                print("✅ Verification passed: Data consistency confirmed")
            else:
                print("❌ Verification failed: Data inconsistency detected")
                
        except Exception as e:
            print(f"❌ Verification failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Convert binary files to NPZ format for DynamicKClusterDistTransformer")
    
    # Required parameters
    parser.add_argument("--input_bin", type=str, required=True,
                       help="Input binary file path")
    parser.add_argument("--output_npz", type=str, required=True,
                       help="Output NPZ file path")
    
    # Format-related parameters
    parser.add_argument("--format", type=str, default="auto",
                       choices=["auto", "fvecs", "bvecs", "raw", "custom"],
                       help="Binary file format (default: auto-detect)")
    parser.add_argument("--dtype", type=str, default="float32",
                       help="Data type for raw/custom formats (default: float32)")
    parser.add_argument("--shape", nargs=2, type=int, metavar=("N", "D"),
                       help="Data shape (N vectors, D dimensions) for raw/custom formats")
    parser.add_argument("--header_size", type=int, default=0,
                       help="Header size in bytes to skip (for custom format)")
    
    # Output-related parameters
    parser.add_argument("--array_name", type=str, default="queries",
                       help="Array name in NPZ file (default: queries)")
    parser.add_argument("--max_vectors", type=int, default=None,
                       help="Maximum number of vectors to convert (for testing)")
    parser.add_argument("--no_verify", action="store_true",
                       help="Skip output file verification")
    
    args = parser.parse_args()
    
    # Process shape parameter
    shape = None
    if args.shape:
        shape = tuple(args.shape)
    
    try:
        convert_bin_to_npz(
            input_bin_path=args.input_bin,
            output_npz_path=args.output_npz,
            format_type=args.format,
            dtype=args.dtype,
            shape=shape,
            header_size=args.header_size,
            max_vectors=args.max_vectors,
            array_name=args.array_name,
            verify=not args.no_verify
        )
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())


'''
# siftsmall_learn.fvecs
cd /home/xln/PycharmProjects/PredictLeafNode/
python src/model/dynamic_k/convert_bin_to_npz.py \
    --input_bin input/Training_data/siftsmall/siftsmall5H/siftsmall_learn.fvecs \
    --output_npz input/Training_data/siftsmall/siftsmall5H/siftsmall_learn.npz \
    --format fvecs


# siftsmall_query.fvecs
cd /home/xln/PycharmProjects/PredictLeafNode/
python src/model/dynamic_k/convert_bin_to_npz.py \
    --input_bin input/Training_data/siftsmall/siftsmall5H/siftsmall_query.fvecs \
    --output_npz input/Training_data/siftsmall/siftsmall5H/siftsmall_query.npz \
    --format fvecs

'''