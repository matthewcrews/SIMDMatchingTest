
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
            0, 2
            3, 5
            12, 14
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
    
    let leftCompactShuffleMasks =
        [|
            Vector128.Create (0x80uy, 0x80uy, 0x80uy, 0x80uy, 0x80uy, 0x80uy, 0x80uy, 0x80uy, 0x80uy, 0x80uy, 0x80uy, 0x80uy, 0x80uy, 0x80uy, 0x80uy, 0x80uy)
            Vector128.Create (0x0uy, 0x1uy, 0x2uy, 0x3uy, 0x80uy, 0x80uy, 0x80uy, 0x80uy, 0x80uy, 0x80uy, 0x80uy, 0x80uy, 0x80uy, 0x80uy, 0x80uy, 0x80uy)
            
            Vector128.Create (0x0uy, 0x1uy, 0x2uy, 0x3uy, 0x4uy, 0x5uy, 0x6uy, 0x7uy, 0x8uy, 0x9uy, 0xauy, 0xbuy, 0xcuy, 0xduy, 0xeuy, 0xfuy)
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
        let nonNegativeCheck : Vector128<float32> = retype nonNegativeCheck
        
        // Compute the MoveMask to lookup Left-Compacting shuffle mask
        let moveMask = Avx2.MoveMask nonNegativeCheck
        // Lookup the Left-Compacting shuffle mask we will need
        let shuffleMask = leftCompactShuffleMasks[moveMask]
        
        // Retype moveMask to use it with PopCount to get number of matches
        let moveMask : uint32 = retype moveMask
        let numberOfMatches = BitOperations.PopCount moveMask
        
        // Retype newStarts and newBounds for shuffling
        let newStarts : Vector128<byte> = retype newStarts
        let newBounds : Vector128<byte> = retype newBounds
        
        // Shuffle the values that we want to keep
        let newStartsPacked : Vector128<int> = retype (Avx2.Shuffle (newStarts, shuffleMask))
        let newBoundsPacked : Vector128<int> = retype (Avx2.Shuffle (newBounds, shuffleMask))
        
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