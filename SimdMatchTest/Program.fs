
open System
open System.Buffers
open System.Numerics
open System.Runtime.Intrinsics
open System.Runtime.Intrinsics.X86
open FSharp.NativeInterop

#nowarn "9"
#nowarn "42"
#nowarn "51"

module Logic =
    
    let inline retype<'T,'U> (x: 'T) : 'U = (# "" x: 'U #)
    
    let aStarts, aBounds =
        let data = [|
            2, 10
        |]        
        let aStarts = data |> Array.map fst
        let aBounds = data |> Array.map snd
        aStarts, aBounds
        
    let bStarts, bBounds =
        let data = [|
            1, 2
            2, 3
            4, 7
            9, 12
            13, 15
            17, 22
            25, 29
            31, 33
            35, 38
        |]
        let bStarts = data |> Array.map fst
        let bBounds = data |> Array.map snd
        bStarts, bBounds
    
    let leftCompactShuffleMasks : Vector128<byte>[] =
        // NOTE: Remember x86 is little-endian therefore we need to select
        let remv = 0x80_80_80_80 // Remove values
        let pos0 = 0x03_02_01_00 // 0th position
        let pos1 = 0x07_06_05_04 // 1st position
        let pos2 = 0x0B_0A_09_08 // 2nd position
        let pos3 = 0x0F_0E_0D_0C // 3rd position
        
        retype [|
            Vector128.Create (remv, remv, remv, remv) // BitMask Pattern: 0000
            Vector128.Create (pos0, remv, remv, remv) // BitMask Pattern: 0001
            Vector128.Create (remv, pos0, remv, remv) // BitMask Pattern: 0010
            Vector128.Create (pos0, pos1, remv, remv) // BitMask Pattern: 0011
            Vector128.Create (remv, remv, pos0, remv) // BitMask Pattern: 0100
            Vector128.Create (pos0, remv, pos0, remv) // BitMask Pattern: 0101
            Vector128.Create (remv, pos0, pos1, remv) // BitMask Pattern: 0110
            Vector128.Create (pos0, pos1, pos2, remv) // BitMask Pattern: 0111
            Vector128.Create (remv, remv, remv, pos0) // BitMask Pattern: 1000
            Vector128.Create (pos0, remv, remv, pos1) // BitMask Pattern: 1001
            Vector128.Create (remv, pos0, remv, pos1) // BitMask Pattern: 1010
            Vector128.Create (pos0, pos1, remv, pos2) // BitMask Pattern: 1011
            Vector128.Create (remv, remv, pos0, pos1) // BitMask Pattern: 1100
            Vector128.Create (pos0, remv, pos1, pos2) // BitMask Pattern: 1101
            Vector128.Create (remv, pos0, pos1, pos2) // BitMask Pattern: 1110
            Vector128.Create (pos0, pos1, pos2, pos3) // BitMask Pattern: 1111
            
        |]
    
    let test () =

        // Accumulator data
        let accStarts = Array.create (aStarts.Length + bStarts.Length) 0
        let accBounds = Array.create (aBounds.Length + bBounds.Length) 0
        let mutable accIdx = 0
        let mutable aIdx = 0
        let mutable bIdx = 0
        
        let aStartsPtr = && aStarts.AsSpan().GetPinnableReference()
        let aBoundsPtr = && aBounds.AsSpan().GetPinnableReference()
        let bStartsPtr = && bStarts.AsSpan().GetPinnableReference()
        let bBoundsPtr = && bBounds.AsSpan().GetPinnableReference()
        let accStartsPtr = && accStarts.AsSpan().GetPinnableReference()
        let accBoundsPtr = && accBounds.AsSpan().GetPinnableReference()
        
        // Load the data into the SIMD registers
        let aStartVec = Avx2.BroadcastScalarToVector128 (NativePtr.add aStartsPtr aIdx)
        let aBoundVec = Avx2.BroadcastScalarToVector128 (NativePtr.add aBoundsPtr aIdx)
        let bStartsVec = Avx2.LoadVector128 (NativePtr.add bStartsPtr bIdx)
        let bBoundsVec = Avx2.LoadVector128 (NativePtr.add bBoundsPtr bIdx)
        
        // Compute new Starts and Bounds
        let newStarts = Avx2.Max (aStartVec, bStartsVec)
        let newBounds = Avx2.Min (aBoundVec, bBoundsVec)
        
        // Perform comparison to check for valid intervals
        let nonNegativeCheck = Avx2.CompareGreaterThan (newBounds, newStarts)
        
        // Retype so we can use MoveMask
        let nonNegativeCheckAsFloat32 : Vector128<float32> = retype nonNegativeCheck
        
        // Compute the MoveMask to lookup Left-Compacting shuffle mask
        let moveMask = Avx2.MoveMask nonNegativeCheckAsFloat32
        // Lookup the Left-Compacting shuffle mask we will need
        let shuffleMask = leftCompactShuffleMasks[moveMask]
        
        // Retype moveMask to use it with PopCount to get number of matches
        let moveMask : uint32 = retype moveMask
        let numberOfMatches = BitOperations.PopCount moveMask
        
        // Retype newStarts and newBounds for shuffling
        let newStartsAsBytes : Vector128<byte> = retype newStarts
        let newBoundsAsBytes : Vector128<byte> = retype newBounds
        
        // Shuffle the values that we want to keep
        let newStartsPacked : Vector128<int> = retype (Avx2.Shuffle (newStartsAsBytes, shuffleMask))
        let newBoundsPacked : Vector128<int> = retype (Avx2.Shuffle (newBoundsAsBytes, shuffleMask))
        
        // Write the values out to the acc arrays
        Avx2.Store (NativePtr.add accStartsPtr accIdx, newStartsPacked)
        Avx2.Store (NativePtr.add accBoundsPtr accIdx, newBoundsPacked)
        
        // Move the accIdx forward so that we write new matches to the correct spot
        accIdx <- accIdx + numberOfMatches
        
        
        ()
        


[<EntryPoint>]
let main (args: string[]) =
    Logic.test()
    1