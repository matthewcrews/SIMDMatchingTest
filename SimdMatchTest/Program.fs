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
            26, 28
            30, 37
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
        // Also note, we are using this with Shuffle, which should be thought
        // of as a Selector of the elements, not masking elements off.
        let zero = 0x80_80_80_80 // Zero
        let elm0 = 0x03_02_01_00 // 0th position
        let elm1 = 0x07_06_05_04 // 1st position
        let elm2 = 0x0B_0A_09_08 // 2nd position
        let elm3 = 0x0F_0E_0D_0C // 3rd position
        
        retype [|
            Vector128.Create (zero, zero, zero, zero) // BitMask Pattern: 0000
            Vector128.Create (elm0, zero, zero, zero) // BitMask Pattern: 0001
            Vector128.Create (elm1, zero, zero, zero) // BitMask Pattern: 0010
            Vector128.Create (elm0, elm1, zero, zero) // BitMask Pattern: 0011
            Vector128.Create (elm2, zero, zero, zero) // BitMask Pattern: 0100
            Vector128.Create (elm0, elm2, zero, zero) // BitMask Pattern: 0101
            Vector128.Create (elm1, elm2, zero, zero) // BitMask Pattern: 0110
            Vector128.Create (elm0, elm1, elm2, zero) // BitMask Pattern: 0111
            Vector128.Create (elm3, zero, zero, zero) // BitMask Pattern: 1000
            Vector128.Create (elm0, elm3, zero, zero) // BitMask Pattern: 1001
            Vector128.Create (elm1, elm3, zero, zero) // BitMask Pattern: 1010
            Vector128.Create (elm0, elm1, elm3, zero) // BitMask Pattern: 1011
            Vector128.Create (elm2, elm3, zero, zero) // BitMask Pattern: 1100
            Vector128.Create (elm0, elm2, elm3, zero) // BitMask Pattern: 1101
            Vector128.Create (elm1, elm2, elm3, zero) // BitMask Pattern: 1110
            Vector128.Create (elm0, elm1, elm2, elm3) // BitMask Pattern: 1111
        |]
    
    let test () =

        // Accumulator data
        let accStarts = Array.create (aStarts.Length + bStarts.Length) 0
        let accBounds = Array.create (aBounds.Length + bBounds.Length) 0
        let mutable accIdx = 0
        let mutable aIdx = 0
        let mutable bIdx = 0
        
        // Only want to perform this loop if Avx2 is supported
        if Avx2.IsSupported then
            
            let lastBlockIdx = Vector128<int>.Count * (bStarts.Length / Vector128<int>.Count)
            let bStartsPtr = && bStarts.AsSpan().GetPinnableReference()
            let bBoundsPtr = && bBounds.AsSpan().GetPinnableReference()
            let accStartsPtr = && accStarts.AsSpan().GetPinnableReference()
            let accBoundsPtr = && accBounds.AsSpan().GetPinnableReference()
            
            while aIdx < aStarts.Length && bIdx < lastBlockIdx do

                // Load the data into the SIMD registers
                let aStartVec = Vector128.Create aStarts[aIdx]
                let aBoundVec = Vector128.Create aBounds[aIdx]
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
                if aBounds[aIdx] < bBounds[bIdx + Vector128<int>.Count - 1] then
                    aIdx <- aIdx + 1
                else
                    bIdx <- bIdx + Vector128<int>.Count
            
        while aIdx < aStarts.Length && bIdx < bStarts.Length do
            
            if aBounds[aIdx] < bStarts[bIdx] then
                aIdx <- aIdx + 1
            elif bBounds[bIdx] < aStarts[aIdx] then
                bIdx <- bIdx + 1
            else
                accStarts[accIdx] <- Math.Max (aStarts[aIdx], bStarts[bIdx])
                accBounds[accIdx] <- Math.Min (aBounds[aIdx], bBounds[bIdx])
                accIdx <- accIdx + 1
                
                if aBounds[aIdx] < bBounds[bIdx] then
                    aIdx <- aIdx + 1
                else
                    bIdx <- bIdx + 1
                    
        let resStarts = GC.AllocateUninitializedArray accIdx
        let resBounds = GC.AllocateUninitializedArray accIdx
            
        Array.Copy (accStarts, resStarts, accIdx)
        Array.Copy (accBounds, resBounds, accIdx)
        
        resStarts, resBounds
        


[<EntryPoint>]
let main (args: string[]) =
    let _ = Logic.test()
    1