// dear imgui: Renderer Backend for Metal
// This needs to be used along with a Platform Backend (e.g. OSX)

// Implemented features:
//  [X] Renderer: User texture binding. Use 'MTLTexture' as texture identifier. Read the FAQ about ImTextureID/ImTextureRef!
//  [X] Renderer: Large meshes support (64k+ vertices) even with 16-bit indices (ImGuiBackendFlags_RendererHasVtxOffset).
//  [X] Renderer: Texture updates support for dynamic font atlas (ImGuiBackendFlags_RendererHasTextures).

// You can use unmodified imgui_impl_* files in your project. See examples/ folder for examples of using this.
// Prefer including the entire imgui/ repository into your project (either as a copy or as a submodule), and only build the backends you need.
// Learn about Dear ImGui:
// - FAQ                  https://dearimgui.com/faq
// - Getting Started      https://dearimgui.com/getting-started
// - Documentation        https://dearimgui.com/docs (same as your local docs/ folder).
// - Introduction, links and more at the top of imgui.cpp

// CHANGELOG
// (minor and older changes stripped away, please see git history for details)
//  2025-09-18: Call platform_io.ClearRendererHandlers() on shutdown.
//  2025-06-11: Added support for ImGuiBackendFlags_RendererHasTextures, for dynamic font atlas. Removed ImGui_ImplMetal_CreateFontsTexture() and ImGui_ImplMetal_DestroyFontsTexture().
//  2025-02-03: Metal: Crash fix. (#8367)
//  2025-01-08: Metal: Fixed memory leaks when using metal-cpp (#8276, #8166) or when using multiple contexts (#7419).
//  2022-08-23: Metal: Update deprecated property 'sampleCount'->'rasterSampleCount'.
//  2022-07-05: Metal: Add dispatch synchronization.
//  2022-06-30: Metal: Use __bridge for ARC based systems.
//  2022-06-01: Metal: Fixed null dereference on exit inside command buffer completion handler.
//  2022-04-27: Misc: Store backend data in a per-context struct, allowing to use this backend with multiple contexts.
//  2022-01-03: Metal: Ignore ImDrawCmd where ElemCount == 0 (very rare but can technically be manufactured by user code).
//  2021-12-30: Metal: Added Metal C++ support. Enable with '#define IMGUI_IMPL_METAL_CPP' in your imconfig.h file.
//  2021-08-24: Metal: Fixed a crash when clipping rect larger than framebuffer is submitted. (#4464)
//  2021-05-19: Metal: Replaced direct access to ImDrawCmd::TextureId with a call to ImDrawCmd::GetTexID(). (will become a requirement)
//  2021-02-18: Metal: Change blending equation to preserve alpha in output buffer.
//  2021-01-25: Metal: Fixed texture storage mode when building on Mac Catalyst.
//  2019-05-29: Metal: Added support for large mesh (64K+ vertices), enable ImGuiBackendFlags_RendererHasVtxOffset flag.
//  2019-04-30: Metal: Added support for special ImDrawCallback_ResetRenderState callback to reset render state.
//  2019-02-11: Metal: Projecting clipping rectangles correctly using draw_data->FramebufferScale to allow multi-viewports for retina display.
//  2018-11-30: Misc: Setting up io.BackendRendererName so it can be displayed in the About Window.
//  2018-07-05: Metal: Added new Metal backend implementation.

#include "imgui.h"
#ifndef IMGUI_DISABLE
#include "imgui_impl_metal.h"
#import <time.h>
#import <Metal/Metal.h>

#pragma mark - Support classes

// A wrapper around a MTLBuffer object that knows the last time it was reused
@interface MetalBuffer : NSObject
@property (nonatomic, strong) id<MTLBuffer> buffer;
@property (nonatomic, assign) double        lastReuseTime;
- (instancetype)initWithBuffer:(id<MTLBuffer>)buffer;
@end

// An object that encapsulates the data necessary to uniquely identify a
// render pipeline state. These are used as cache keys.
@interface FramebufferDescriptor : NSObject<NSCopying>
@property (nonatomic, assign) unsigned long  sampleCount;
@property (nonatomic, assign) MTLPixelFormat colorPixelFormat;
@property (nonatomic, assign) MTLPixelFormat depthPixelFormat;
@property (nonatomic, assign) MTLPixelFormat stencilPixelFormat;
- (instancetype)initWithRenderPassDescriptor:(MTLRenderPassDescriptor*)renderPassDescriptor;
#if defined(__MAC_26_0) || defined(__IPHONE_26_0)
- (instancetype)initWithRenderPassDescriptor4:(MTL4RenderPassDescriptor*)renderPassDescriptor;
#endif
@end

@interface MetalTexture : NSObject
@property (nonatomic, strong) id<MTLTexture> metalTexture;
- (instancetype)initWithTexture:(id<MTLTexture>)metalTexture;
@end

#if defined(__MAC_26_0) || defined(__IPHONE_26_0)
@interface MetalFrameResources : NSObject
@property (nonatomic, strong) id<MTLBuffer> vertexBuffer;
@property (nonatomic, strong) id<MTLBuffer> indexBuffer;
@property (nonatomic, strong) id<MTLBuffer> uniformBuffer;
@property (nonatomic, strong) NSMutableArray<id<MTL4ArgumentTable>>* argumentTables;
@property (nonatomic, assign) NSUInteger argumentTableCursor;
@end
#endif

// A singleton that stores long-lived objects that are needed by the Metal
// renderer backend. Stores the render pipeline state cache and the default
// font texture, and manages the reusable buffer cache.
@interface MetalContext : NSObject
@property (nonatomic, strong) id<MTLDevice>                 device;
@property (nonatomic, strong) id<MTLDepthStencilState>      depthStencilState;
@property (nonatomic, strong) FramebufferDescriptor*        framebufferDescriptor; // framebuffer descriptor for current frame; transient
@property (nonatomic, strong) NSMutableDictionary*          renderPipelineStateCache; // pipeline cache; keyed on framebuffer descriptors
#if defined(__MAC_26_0) || defined(__IPHONE_26_0)
@property (nonatomic, strong) NSMutableDictionary*          metal4RenderPipelineStateCache; // Metal4-specific pipeline cache
#endif
@property (nonatomic, strong) NSMutableArray<MetalBuffer*>* bufferCache;
#if defined(__MAC_26_0) || defined(__IPHONE_26_0)
@property (nonatomic, strong) MTL4ArgumentTableDescriptor*  renderArgumentTableDescriptor;
@property (nonatomic, strong) id<MTLResidencySet>           metal4ResidencySet;
@property (nonatomic, strong) id<MTLSharedEvent>            metal4SharedEvent;
@property (nonatomic, strong) NSMutableArray<MetalFrameResources*>* metal4FrameResources;
@property (nonatomic, assign) NSUInteger                    metal4MaxFramesInFlight;
@property (nonatomic, assign) NSUInteger                    metal4FrameIndex;
@property (nonatomic, assign) uint64_t                      metal4FrameNumber;
@property (nonatomic, assign) BOOL                          metal4CanReuseFrameResources;
#endif
@property (nonatomic, assign) double                        lastBufferCachePurge;
- (MetalBuffer*)dequeueReusableBufferOfLength:(NSUInteger)length device:(id<MTLDevice>)device;
- (id<MTLRenderPipelineState>)renderPipelineStateForFramebufferDescriptor:(FramebufferDescriptor*)descriptor device:(id<MTLDevice>)device;
#if defined(__MAC_26_0) || defined(__IPHONE_26_0)
- (id<MTLRenderPipelineState>)renderPipelineStateForFramebufferDescriptor4:(FramebufferDescriptor*)descriptor device:(id<MTLDevice>)device;
#endif
@end

struct ImGui_ImplMetal_Data
{
    MetalContext*               SharedMetalContext;

    ImGui_ImplMetal_Data()      { memset((void*)this, 0, sizeof(*this)); }
};

static ImGui_ImplMetal_Data*    ImGui_ImplMetal_GetBackendData()    { return ImGui::GetCurrentContext() ? (ImGui_ImplMetal_Data*)ImGui::GetIO().BackendRendererUserData : nullptr; }
static void                     ImGui_ImplMetal_DestroyBackendData(){ IM_DELETE(ImGui_ImplMetal_GetBackendData()); }

static inline CFTimeInterval    GetMachAbsoluteTimeInSeconds()      { return (CFTimeInterval)(double)(clock_gettime_nsec_np(CLOCK_UPTIME_RAW) / 1e9); }

#if defined(__MAC_26_0) || defined(__IPHONE_26_0)
static const NSUInteger         MetalArgumentBufferIndex_VertexData = 0;
static const NSUInteger         MetalArgumentBufferIndex_Uniforms = 1;
static const NSUInteger         MetalArgumentBufferIndex_IndexData = 2;
static const NSUInteger         MetalArgumentTextureIndex_FontAtlas = 0;
static const uint64_t           Metal4FrameWaitTimeoutMS = 1000;
#endif

#ifdef IMGUI_IMPL_METAL_CPP

#pragma mark - Dear ImGui Metal C++ Backend API

bool ImGui_ImplMetal_Init(MTL::Device* device)
{
    return ImGui_ImplMetal_Init((__bridge id<MTLDevice>)(device));
}

void ImGui_ImplMetal_NewFrame(MTL::RenderPassDescriptor* renderPassDescriptor)
{
    ImGui_ImplMetal_NewFrame((__bridge MTLRenderPassDescriptor*)(renderPassDescriptor));
}

void ImGui_ImplMetal_RenderDrawData(ImDrawData* draw_data,
                                    MTL::CommandBuffer* commandBuffer,
                                    MTL::RenderCommandEncoder* commandEncoder)
{
    ImGui_ImplMetal_RenderDrawData(draw_data,
                                   (__bridge id<MTLCommandBuffer>)(commandBuffer),
                                   (__bridge id<MTLRenderCommandEncoder>)(commandEncoder));

}

bool ImGui_ImplMetal_CreateDeviceObjects(MTL::Device* device)
{
    return ImGui_ImplMetal_CreateDeviceObjects((__bridge id<MTLDevice>)(device));
}

#endif // #ifdef IMGUI_IMPL_METAL_CPP

#pragma mark - Dear ImGui Metal Backend API

bool ImGui_ImplMetal_Init(id<MTLDevice> device)
{
    ImGuiIO& io = ImGui::GetIO();
    IMGUI_CHECKVERSION();
    IM_ASSERT(io.BackendRendererUserData == nullptr && "Already initialized a renderer backend!");

    ImGui_ImplMetal_Data* bd = IM_NEW(ImGui_ImplMetal_Data)();
    io.BackendRendererUserData = (void*)bd;
    io.BackendRendererName = "imgui_impl_metal";
    io.BackendFlags |= ImGuiBackendFlags_RendererHasVtxOffset;  // We can honor the ImDrawCmd::VtxOffset field, allowing for large meshes.
    io.BackendFlags |= ImGuiBackendFlags_RendererHasTextures;   // We can honor ImGuiPlatformIO::Textures[] requests during render.

    bd->SharedMetalContext = [[MetalContext alloc] init];
    bd->SharedMetalContext.device = device;

    return true;
}

void ImGui_ImplMetal_Shutdown()
{
    ImGui_ImplMetal_Data* bd = ImGui_ImplMetal_GetBackendData();
    IM_UNUSED(bd);
    IM_ASSERT(bd != nullptr && "No renderer backend to shutdown, or already shutdown?");
    ImGuiIO& io = ImGui::GetIO();
    ImGuiPlatformIO& platform_io = ImGui::GetPlatformIO();

    ImGui_ImplMetal_DestroyDeviceObjects();
    ImGui_ImplMetal_DestroyBackendData();

    io.BackendRendererName = nullptr;
    io.BackendRendererUserData = nullptr;
    io.BackendFlags &= ~(ImGuiBackendFlags_RendererHasVtxOffset | ImGuiBackendFlags_RendererHasTextures);
    platform_io.ClearRendererHandlers();
}

void ImGui_ImplMetal_NewFrame(MTLRenderPassDescriptor* renderPassDescriptor)
{
    ImGui_ImplMetal_Data* bd = ImGui_ImplMetal_GetBackendData();
    IM_ASSERT(bd != nil && "Context or backend not initialized! Did you call ImGui_ImplMetal_Init()?");
#ifdef IMGUI_IMPL_METAL_CPP
    bd->SharedMetalContext.framebufferDescriptor = [[[FramebufferDescriptor alloc] initWithRenderPassDescriptor:renderPassDescriptor]autorelease];
#else
    bd->SharedMetalContext.framebufferDescriptor = [[FramebufferDescriptor alloc] initWithRenderPassDescriptor:renderPassDescriptor];
#endif
    if (bd->SharedMetalContext.depthStencilState == nil)
        ImGui_ImplMetal_CreateDeviceObjects(bd->SharedMetalContext.device);
}

#if defined(__MAC_26_0) || defined(__IPHONE_26_0)
void ImGui_ImplMetal4_ConfigureFrameSynchronization(id<MTLSharedEvent> sharedEvent, int maxFramesInFlight)
{
    ImGui_ImplMetal_Data* bd = ImGui_ImplMetal_GetBackendData();
    IM_ASSERT(bd != nil && "Context or backend not initialized! Did you call ImGui_ImplMetal_Init()?");
    IM_ASSERT(maxFramesInFlight >= 0);

    MetalContext* ctx = bd->SharedMetalContext;
    ctx.metal4SharedEvent = sharedEvent;
    ctx.metal4MaxFramesInFlight = (NSUInteger)maxFramesInFlight;
    ctx.metal4FrameIndex = 0;
    ctx.metal4FrameNumber = 0;
    ctx.metal4CanReuseFrameResources = YES;
    ctx.metal4FrameResources = [NSMutableArray array];
    for (NSUInteger frame_index = 0; frame_index < ctx.metal4MaxFramesInFlight; frame_index++)
    {
        MetalFrameResources* frame_resources = [[MetalFrameResources alloc] init];
        frame_resources.argumentTables = [NSMutableArray array];
        frame_resources.argumentTableCursor = 0;
        [ctx.metal4FrameResources addObject:frame_resources];
    }
}

void ImGui_ImplMetal4_NotifyFrameSubmitted(id<MTL4CommandQueue> commandQueue)
{
    ImGui_ImplMetal_Data* bd = ImGui_ImplMetal_GetBackendData();
    IM_ASSERT(bd != nil && "Context or backend not initialized! Did you call ImGui_ImplMetal_Init()?");
    MetalContext* ctx = bd->SharedMetalContext;
    if (ctx.metal4SharedEvent == nil || ctx.metal4MaxFramesInFlight == 0)
        return;

    [commandQueue signalEvent:ctx.metal4SharedEvent value:ctx.metal4FrameNumber];
}

void ImGui_ImplMetal4_NewFrame(MTL4RenderPassDescriptor* renderPassDescriptor)
{
    ImGui_ImplMetal_Data* bd = ImGui_ImplMetal_GetBackendData();
    IM_ASSERT(bd != nil && "Context or backend not initialized! Did you call ImGui_ImplMetal_Init()?");
    MetalContext* ctx = bd->SharedMetalContext;
    if (ctx.metal4SharedEvent != nil && ctx.metal4MaxFramesInFlight > 0)
    {
        ctx.metal4FrameNumber += 1;
        ctx.metal4FrameIndex = (NSUInteger)((ctx.metal4FrameNumber - 1) % ctx.metal4MaxFramesInFlight);
        ctx.metal4CanReuseFrameResources = YES;
        if (ctx.metal4FrameNumber > ctx.metal4MaxFramesInFlight)
        {
            uint64_t completed_frame = ctx.metal4FrameNumber - ctx.metal4MaxFramesInFlight;
            BOOL signaled = [ctx.metal4SharedEvent waitUntilSignaledValue:completed_frame timeoutMS:Metal4FrameWaitTimeoutMS];
            if (!signaled)
            {
                NSLog(@"ImGui_ImplMetal4: timed out waiting for frame %llu", completed_frame);
                ctx.metal4CanReuseFrameResources = NO;
            }
        }
        if (ctx.metal4CanReuseFrameResources && ctx.metal4FrameResources.count == ctx.metal4MaxFramesInFlight)
            ctx.metal4FrameResources[ctx.metal4FrameIndex].argumentTableCursor = 0;
    }

    bd->SharedMetalContext.framebufferDescriptor = [[FramebufferDescriptor alloc] initWithRenderPassDescriptor4:renderPassDescriptor];
    if (bd->SharedMetalContext.depthStencilState == nil)
        ImGui_ImplMetal_CreateDeviceObjects(bd->SharedMetalContext.device);
}
#endif

static void ImGui_ImplMetal_SetupRenderState(ImDrawData* draw_data, id<MTLCommandBuffer> commandBuffer,
    id<MTLRenderCommandEncoder> commandEncoder, id<MTLRenderPipelineState> renderPipelineState,
    MetalBuffer* vertexBuffer, size_t vertexBufferOffset)
{
    IM_UNUSED(commandBuffer);
    ImGui_ImplMetal_Data* bd = ImGui_ImplMetal_GetBackendData();
    [commandEncoder setCullMode:MTLCullModeNone];
    [commandEncoder setDepthStencilState:bd->SharedMetalContext.depthStencilState];

    // Setup viewport, orthographic projection matrix
    // Our visible imgui space lies from draw_data->DisplayPos (top left) to
    // draw_data->DisplayPos+data_data->DisplaySize (bottom right). DisplayMin is typically (0,0) for single viewport apps.
    MTLViewport viewport =
    {
        .originX = 0.0,
        .originY = 0.0,
        .width = (double)(draw_data->DisplaySize.x * draw_data->FramebufferScale.x),
        .height = (double)(draw_data->DisplaySize.y * draw_data->FramebufferScale.y),
        .znear = 0.0,
        .zfar = 1.0
    };
    [commandEncoder setViewport:viewport];

    float L = draw_data->DisplayPos.x;
    float R = draw_data->DisplayPos.x + draw_data->DisplaySize.x;
    float T = draw_data->DisplayPos.y;
    float B = draw_data->DisplayPos.y + draw_data->DisplaySize.y;
    float N = (float)viewport.znear;
    float F = (float)viewport.zfar;
    const float ortho_projection[4][4] =
    {
        { 2.0f/(R-L),   0.0f,           0.0f,   0.0f },
        { 0.0f,         2.0f/(T-B),     0.0f,   0.0f },
        { 0.0f,         0.0f,        1/(F-N),   0.0f },
        { (R+L)/(L-R),  (T+B)/(B-T), N/(F-N),   1.0f },
    };
    [commandEncoder setVertexBytes:&ortho_projection length:sizeof(ortho_projection) atIndex:1];

    [commandEncoder setRenderPipelineState:renderPipelineState];

    [commandEncoder setVertexBuffer:vertexBuffer.buffer offset:0 atIndex:0];
    [commandEncoder setVertexBufferOffset:vertexBufferOffset atIndex:0];
}

#if defined(__MAC_26_0) || defined(__IPHONE_26_0)
static void ImGui_ImplMetal4_SetupRenderState(ImDrawData* draw_data,
    id<MTL4RenderCommandEncoder> commandEncoder, id<MTLRenderPipelineState> renderPipelineState)
{
    ImGui_ImplMetal_Data* bd = ImGui_ImplMetal_GetBackendData();
    [commandEncoder setCullMode:MTLCullModeNone];
    [commandEncoder setDepthStencilState:bd->SharedMetalContext.depthStencilState];

    MTLViewport viewport =
    {
        .originX = 0.0,
        .originY = 0.0,
        .width = (double)(draw_data->DisplaySize.x * draw_data->FramebufferScale.x),
        .height = (double)(draw_data->DisplaySize.y * draw_data->FramebufferScale.y),
        .znear = 0.0,
        .zfar = 1.0
    };
    [commandEncoder setViewport:viewport];
    [commandEncoder setRenderPipelineState:renderPipelineState];
}

static void ImGui_ImplMetal4_SetupRenderResources(id<MTL4ArgumentTable> argumentTable,
    id<MTLBuffer> vertexBuffer, size_t vertexBufferOffset,
    id<MTLBuffer> uniformBuffer,
    id<MTLBuffer> indexBuffer, size_t indexBufferOffset)
{
    [argumentTable setAddress:vertexBuffer.gpuAddress + vertexBufferOffset
                      atIndex:MetalArgumentBufferIndex_VertexData];
    [argumentTable setAddress:uniformBuffer.gpuAddress atIndex:MetalArgumentBufferIndex_Uniforms];
    [argumentTable setAddress:indexBuffer.gpuAddress + indexBufferOffset atIndex:MetalArgumentBufferIndex_IndexData];
}

static id<MTLBuffer> ImGui_ImplMetal4_EnsureBuffer(id<MTLDevice> device, id<MTLBuffer> buffer, size_t length)
{
    if (buffer == nil || buffer.length < length)
        return [device newBufferWithLength:length options:MTLResourceStorageModeShared];
    return buffer;
}

static void ImGui_ImplMetal4_ResidencySetAddAllocation(id<MTLResidencySet> residencySet, id<MTLAllocation> allocation)
{
    if (residencySet != nil && allocation != nil && ![residencySet containsAllocation:allocation])
    {
        [residencySet addAllocation:allocation];
        [residencySet commit];
    }
}

static void ImGui_ImplMetal4_ResidencySetRemoveAllocation(id<MTLResidencySet> residencySet, id<MTLAllocation> allocation)
{
    if (residencySet != nil && allocation != nil && [residencySet containsAllocation:allocation])
    {
        [residencySet removeAllocation:allocation];
        [residencySet commit];
    }
}

static id<MTL4ArgumentTable> ImGui_ImplMetal4_NewArgumentTable(id<MTLDevice> device, MTL4ArgumentTableDescriptor* descriptor)
{
    if (device == nil || descriptor == nil)
        return nil;
    NSError* error = nil;
    id<MTL4ArgumentTable> argumentTable = [device newArgumentTableWithDescriptor:descriptor error:&error];
    if (error != nil)
        NSLog(@"Error: failed to create Metal4 argument table: %@", error);
    return argumentTable;
}

static id<MTL4ArgumentTable> ImGui_ImplMetal4_LeaseArgumentTable(MetalContext* ctx, id<MTLDevice> device)
{
    if (ctx == nil || ctx.renderArgumentTableDescriptor == nil)
        return nil;

    if (ctx.metal4CanReuseFrameResources && ctx.metal4SharedEvent != nil && ctx.metal4MaxFramesInFlight > 0 && ctx.metal4FrameResources.count == ctx.metal4MaxFramesInFlight)
    {
        MetalFrameResources* frame_resources = ctx.metal4FrameResources[ctx.metal4FrameIndex];
        if (frame_resources.argumentTables == nil)
            frame_resources.argumentTables = [NSMutableArray array];

        if (frame_resources.argumentTableCursor >= frame_resources.argumentTables.count)
        {
            id<MTL4ArgumentTable> argumentTable = ImGui_ImplMetal4_NewArgumentTable(device, ctx.renderArgumentTableDescriptor);
            if (argumentTable == nil)
                return nil;
            [frame_resources.argumentTables addObject:argumentTable];
        }

        return frame_resources.argumentTables[frame_resources.argumentTableCursor++];
    }

    return ImGui_ImplMetal4_NewArgumentTable(device, ctx.renderArgumentTableDescriptor);
}
#endif

// Metal Render function.
void ImGui_ImplMetal_RenderDrawData(ImDrawData* draw_data, id<MTLCommandBuffer> commandBuffer, id<MTLRenderCommandEncoder> commandEncoder)
{
    ImGui_ImplMetal_Data* bd = ImGui_ImplMetal_GetBackendData();
    MetalContext* ctx = bd->SharedMetalContext;

    // Avoid rendering when minimized, scale coordinates for retina displays (screen coordinates != framebuffer coordinates)
    int fb_width = (int)(draw_data->DisplaySize.x * draw_data->FramebufferScale.x);
    int fb_height = (int)(draw_data->DisplaySize.y * draw_data->FramebufferScale.y);
    if (fb_width <= 0 || fb_height <= 0 || draw_data->CmdLists.Size == 0)
        return;

    // Catch up with texture updates. Most of the times, the list will have 1 element with an OK status, aka nothing to do.
    // (This almost always points to ImGui::GetPlatformIO().Textures[] but is part of ImDrawData to allow overriding or disabling texture updates).
    if (draw_data->Textures != nullptr)
        for (ImTextureData* tex : *draw_data->Textures)
            if (tex->Status != ImTextureStatus_OK)
                ImGui_ImplMetal_UpdateTexture(tex);

    // Try to retrieve a render pipeline state that is compatible with the framebuffer config for this frame
    // The hit rate for this cache should be very near 100%.
    id<MTLRenderPipelineState> renderPipelineState = ctx.renderPipelineStateCache[ctx.framebufferDescriptor];
    if (renderPipelineState == nil)
    {
        // No luck; make a new render pipeline state
        renderPipelineState = [ctx renderPipelineStateForFramebufferDescriptor:ctx.framebufferDescriptor device:commandBuffer.device];

        // Cache render pipeline state for later reuse
        ctx.renderPipelineStateCache[ctx.framebufferDescriptor] = renderPipelineState;
    }

    size_t vertexBufferLength = (size_t)draw_data->TotalVtxCount * sizeof(ImDrawVert);
    size_t indexBufferLength = (size_t)draw_data->TotalIdxCount * sizeof(ImDrawIdx);
    MetalBuffer* vertexBuffer = [ctx dequeueReusableBufferOfLength:vertexBufferLength device:commandBuffer.device];
    MetalBuffer* indexBuffer = [ctx dequeueReusableBufferOfLength:indexBufferLength device:commandBuffer.device];

    ImGui_ImplMetal_SetupRenderState(draw_data, commandBuffer, commandEncoder, renderPipelineState, vertexBuffer, 0);

    // Will project scissor/clipping rectangles into framebuffer space
    ImVec2 clip_off = draw_data->DisplayPos;         // (0,0) unless using multi-viewports
    ImVec2 clip_scale = draw_data->FramebufferScale; // (1,1) unless using retina display which are often (2,2)

    // Render command lists
    size_t vertexBufferOffset = 0;
    size_t indexBufferOffset = 0;
    for (const ImDrawList* draw_list : draw_data->CmdLists)
    {
        memcpy((char*)vertexBuffer.buffer.contents + vertexBufferOffset, draw_list->VtxBuffer.Data, (size_t)draw_list->VtxBuffer.Size * sizeof(ImDrawVert));
        memcpy((char*)indexBuffer.buffer.contents + indexBufferOffset, draw_list->IdxBuffer.Data, (size_t)draw_list->IdxBuffer.Size * sizeof(ImDrawIdx));

        for (int cmd_i = 0; cmd_i < draw_list->CmdBuffer.Size; cmd_i++)
        {
            const ImDrawCmd* pcmd = &draw_list->CmdBuffer[cmd_i];
            if (pcmd->UserCallback)
            {
                // User callback, registered via ImDrawList::AddCallback()
                // (ImDrawCallback_ResetRenderState is a special callback value used by the user to request the renderer to reset render state.)
                if (pcmd->UserCallback == ImDrawCallback_ResetRenderState)
                    ImGui_ImplMetal_SetupRenderState(draw_data, commandBuffer, commandEncoder, renderPipelineState, vertexBuffer, vertexBufferOffset);
                else
                    pcmd->UserCallback(draw_list, pcmd);
            }
            else
            {
                // Project scissor/clipping rectangles into framebuffer space
                ImVec2 clip_min((pcmd->ClipRect.x - clip_off.x) * clip_scale.x, (pcmd->ClipRect.y - clip_off.y) * clip_scale.y);
                ImVec2 clip_max((pcmd->ClipRect.z - clip_off.x) * clip_scale.x, (pcmd->ClipRect.w - clip_off.y) * clip_scale.y);

                // Clamp to viewport as setScissorRect() won't accept values that are off bounds
                if (clip_min.x < 0.0f) { clip_min.x = 0.0f; }
                if (clip_min.y < 0.0f) { clip_min.y = 0.0f; }
                if (clip_max.x > fb_width) { clip_max.x = (float)fb_width; }
                if (clip_max.y > fb_height) { clip_max.y = (float)fb_height; }
                if (clip_max.x <= clip_min.x || clip_max.y <= clip_min.y)
                    continue;
                if (pcmd->ElemCount == 0) // drawIndexedPrimitives() validation doesn't accept this
                    continue;

                // Apply scissor/clipping rectangle
                MTLScissorRect scissorRect =
                {
                    .x = NSUInteger(clip_min.x),
                    .y = NSUInteger(clip_min.y),
                    .width = NSUInteger(clip_max.x - clip_min.x),
                    .height = NSUInteger(clip_max.y - clip_min.y)
                };
                [commandEncoder setScissorRect:scissorRect];

                // Bind texture, Draw
                if (ImTextureID tex_id = pcmd->GetTexID())
                    [commandEncoder setFragmentTexture:(__bridge id<MTLTexture>)(void*)(intptr_t)(tex_id) atIndex:0];

                [commandEncoder setVertexBufferOffset:(vertexBufferOffset + pcmd->VtxOffset * sizeof(ImDrawVert)) atIndex:0];
                [commandEncoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                                           indexCount:pcmd->ElemCount
                                            indexType:sizeof(ImDrawIdx) == 2 ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32
                                          indexBuffer:indexBuffer.buffer
                                    indexBufferOffset:indexBufferOffset + pcmd->IdxOffset * sizeof(ImDrawIdx)];
            }
        }

        vertexBufferOffset += (size_t)draw_list->VtxBuffer.Size * sizeof(ImDrawVert);
        indexBufferOffset += (size_t)draw_list->IdxBuffer.Size * sizeof(ImDrawIdx);
    }

    MetalContext* sharedMetalContext = bd->SharedMetalContext;
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer>)
    {
        dispatch_async(dispatch_get_main_queue(), ^{
            @synchronized(sharedMetalContext.bufferCache)
            {
                [sharedMetalContext.bufferCache addObject:vertexBuffer];
                [sharedMetalContext.bufferCache addObject:indexBuffer];
            }
        });
    }];
}

#if defined(__MAC_26_0) || defined(__IPHONE_26_0)
void ImGui_ImplMetal4_RenderDrawData(ImDrawData* draw_data, id<MTL4CommandBuffer> commandBuffer, id<MTL4RenderCommandEncoder> commandEncoder)
{
    ImGui_ImplMetal_Data* bd = ImGui_ImplMetal_GetBackendData();
    MetalContext* ctx = bd->SharedMetalContext;

    int fb_width = (int)(draw_data->DisplaySize.x * draw_data->FramebufferScale.x);
    int fb_height = (int)(draw_data->DisplaySize.y * draw_data->FramebufferScale.y);
    if (fb_width <= 0 || fb_height <= 0 || draw_data->CmdLists.Size == 0)
        return;

    if (draw_data->Textures != nullptr)
        for (ImTextureData* tex : *draw_data->Textures)
            if (tex->Status != ImTextureStatus_OK)
                ImGui_ImplMetal_UpdateTexture(tex);

    id<MTLRenderPipelineState> renderPipelineState = ctx.metal4RenderPipelineStateCache[ctx.framebufferDescriptor];
    if (renderPipelineState == nil)
    {
        renderPipelineState = [ctx renderPipelineStateForFramebufferDescriptor4:ctx.framebufferDescriptor device:commandBuffer.device];
        ctx.metal4RenderPipelineStateCache[ctx.framebufferDescriptor] = renderPipelineState;
    }

    size_t vertexBufferLength = (size_t)draw_data->TotalVtxCount * sizeof(ImDrawVert);
    size_t indexBufferLength = (size_t)draw_data->TotalIdxCount * sizeof(ImDrawIdx);
    id<MTLBuffer> vertexBuffer = nil;
    id<MTLBuffer> indexBuffer = nil;
    id<MTLBuffer> uniformBuffer = nil;
    id<MTLResidencySet> transientResidencySet = nil;
    if (ctx.metal4CanReuseFrameResources && ctx.metal4SharedEvent != nil && ctx.metal4MaxFramesInFlight > 0 && ctx.metal4FrameResources.count == ctx.metal4MaxFramesInFlight)
    {
        MetalFrameResources* frame_resources = ctx.metal4FrameResources[ctx.metal4FrameIndex];
        id<MTLBuffer> oldVertexBuffer = frame_resources.vertexBuffer;
        id<MTLBuffer> oldIndexBuffer = frame_resources.indexBuffer;
        id<MTLBuffer> oldUniformBuffer = frame_resources.uniformBuffer;
        frame_resources.vertexBuffer = ImGui_ImplMetal4_EnsureBuffer(commandBuffer.device, frame_resources.vertexBuffer, vertexBufferLength);
        frame_resources.indexBuffer = ImGui_ImplMetal4_EnsureBuffer(commandBuffer.device, frame_resources.indexBuffer, indexBufferLength);
        frame_resources.uniformBuffer = ImGui_ImplMetal4_EnsureBuffer(commandBuffer.device, frame_resources.uniformBuffer, sizeof(float[4][4]));
        if (oldVertexBuffer != nil && oldVertexBuffer != frame_resources.vertexBuffer)
            ImGui_ImplMetal4_ResidencySetRemoveAllocation(ctx.metal4ResidencySet, oldVertexBuffer);
        if (oldIndexBuffer != nil && oldIndexBuffer != frame_resources.indexBuffer)
            ImGui_ImplMetal4_ResidencySetRemoveAllocation(ctx.metal4ResidencySet, oldIndexBuffer);
        if (oldUniformBuffer != nil && oldUniformBuffer != frame_resources.uniformBuffer)
            ImGui_ImplMetal4_ResidencySetRemoveAllocation(ctx.metal4ResidencySet, oldUniformBuffer);
        vertexBuffer = frame_resources.vertexBuffer;
        indexBuffer = frame_resources.indexBuffer;
        uniformBuffer = frame_resources.uniformBuffer;
        ImGui_ImplMetal4_ResidencySetAddAllocation(ctx.metal4ResidencySet, vertexBuffer);
        ImGui_ImplMetal4_ResidencySetAddAllocation(ctx.metal4ResidencySet, indexBuffer);
        ImGui_ImplMetal4_ResidencySetAddAllocation(ctx.metal4ResidencySet, uniformBuffer);
    }
    else
    {
        vertexBuffer = [commandBuffer.device newBufferWithLength:vertexBufferLength options:MTLResourceStorageModeShared];
        indexBuffer = [commandBuffer.device newBufferWithLength:indexBufferLength options:MTLResourceStorageModeShared];
        uniformBuffer = [commandBuffer.device newBufferWithLength:sizeof(float[4][4]) options:MTLResourceStorageModeShared];
        if ([commandBuffer.device respondsToSelector:@selector(newResidencySetWithDescriptor:error:)])
        {
            NSError* error = nil;
            MTLResidencySetDescriptor* residencyDescriptor = [[MTLResidencySetDescriptor alloc] init];
            transientResidencySet = [commandBuffer.device newResidencySetWithDescriptor:residencyDescriptor error:&error];
            if (error != nil)
                NSLog(@"Error: failed to create transient Metal4 residency set: %@", error);
            ImGui_ImplMetal4_ResidencySetAddAllocation(transientResidencySet, vertexBuffer);
            ImGui_ImplMetal4_ResidencySetAddAllocation(transientResidencySet, indexBuffer);
            ImGui_ImplMetal4_ResidencySetAddAllocation(transientResidencySet, uniformBuffer);
        }
    }

    if (ctx.metal4ResidencySet != nil && transientResidencySet != nil)
    {
        id<MTLResidencySet> residencySets[2] = { ctx.metal4ResidencySet, transientResidencySet };
        [commandBuffer useResidencySets:residencySets count:2];
    }
    else if (ctx.metal4ResidencySet != nil)
    {
        [commandBuffer useResidencySet:ctx.metal4ResidencySet];
    }
    else if (transientResidencySet != nil)
    {
        [commandBuffer useResidencySet:transientResidencySet];
    }

    float L = draw_data->DisplayPos.x;
    float R = draw_data->DisplayPos.x + draw_data->DisplaySize.x;
    float T = draw_data->DisplayPos.y;
    float B = draw_data->DisplayPos.y + draw_data->DisplaySize.y;
    const float N = 0.0f;
    const float F = 1.0f;
    const float ortho_projection[4][4] =
    {
        { 2.0f/(R-L),   0.0f,           0.0f,   0.0f },
        { 0.0f,         2.0f/(T-B),     0.0f,   0.0f },
        { 0.0f,         0.0f,        1/(F-N),   0.0f },
        { (R+L)/(L-R),  (T+B)/(B-T), N/(F-N),   1.0f },
    };
    memcpy(uniformBuffer.contents, ortho_projection, sizeof(ortho_projection));

    ImGui_ImplMetal4_SetupRenderState(draw_data, commandEncoder, renderPipelineState);

    ImVec2 clip_off = draw_data->DisplayPos;
    ImVec2 clip_scale = draw_data->FramebufferScale;

    size_t vertexBufferOffset = 0;
    size_t indexBufferOffset = 0;
    for (const ImDrawList* draw_list : draw_data->CmdLists)
    {
        memcpy((char*)vertexBuffer.contents + vertexBufferOffset, draw_list->VtxBuffer.Data, (size_t)draw_list->VtxBuffer.Size * sizeof(ImDrawVert));
        memcpy((char*)indexBuffer.contents + indexBufferOffset, draw_list->IdxBuffer.Data, (size_t)draw_list->IdxBuffer.Size * sizeof(ImDrawIdx));

        for (int cmd_i = 0; cmd_i < draw_list->CmdBuffer.Size; cmd_i++)
        {
            const ImDrawCmd* pcmd = &draw_list->CmdBuffer[cmd_i];
            if (pcmd->UserCallback)
            {
                if (pcmd->UserCallback == ImDrawCallback_ResetRenderState)
                {
                    ImGui_ImplMetal4_SetupRenderState(draw_data, commandEncoder, renderPipelineState);
                }
                else
                {
                    pcmd->UserCallback(draw_list, pcmd);
                }
            }
            else
            {
                ImVec2 clip_min((pcmd->ClipRect.x - clip_off.x) * clip_scale.x, (pcmd->ClipRect.y - clip_off.y) * clip_scale.y);
                ImVec2 clip_max((pcmd->ClipRect.z - clip_off.x) * clip_scale.x, (pcmd->ClipRect.w - clip_off.y) * clip_scale.y);

                if (clip_min.x < 0.0f) { clip_min.x = 0.0f; }
                if (clip_min.y < 0.0f) { clip_min.y = 0.0f; }
                if (clip_max.x > fb_width) { clip_max.x = (float)fb_width; }
                if (clip_max.y > fb_height) { clip_max.y = (float)fb_height; }
                if (clip_max.x <= clip_min.x || clip_max.y <= clip_min.y)
                    continue;
                if (pcmd->ElemCount == 0)
                    continue;

                id<MTL4ArgumentTable> argumentTable = ImGui_ImplMetal4_LeaseArgumentTable(ctx, commandBuffer.device);
                if (argumentTable == nil)
                    continue;
                ImGui_ImplMetal4_SetupRenderResources(argumentTable,
                    vertexBuffer, vertexBufferOffset + pcmd->VtxOffset * sizeof(ImDrawVert),
                    uniformBuffer,
                    indexBuffer, indexBufferOffset + pcmd->IdxOffset * sizeof(ImDrawIdx));

                MTLScissorRect scissorRect =
                {
                    .x = NSUInteger(clip_min.x),
                    .y = NSUInteger(clip_min.y),
                    .width = NSUInteger(clip_max.x - clip_min.x),
                    .height = NSUInteger(clip_max.y - clip_min.y)
                };
                [commandEncoder setScissorRect:scissorRect];

                if (ImTextureID tex_id = pcmd->GetTexID())
                {
                    id<MTLTexture> texture = (__bridge id<MTLTexture>)(void*)(intptr_t)(tex_id);
                    [argumentTable setTexture:texture.gpuResourceID atIndex:MetalArgumentTextureIndex_FontAtlas];
                }
                else
                {
                    [argumentTable setTexture:MTLResourceID{0} atIndex:MetalArgumentTextureIndex_FontAtlas];
                }

                [commandEncoder setArgumentTable:argumentTable atStages:MTLRenderStageVertex | MTLRenderStageFragment];
                [commandEncoder drawPrimitives:MTLPrimitiveTypeTriangle
                                   vertexStart:0
                                   vertexCount:pcmd->ElemCount];
            }
        }

        vertexBufferOffset += (size_t)draw_list->VtxBuffer.Size * sizeof(ImDrawVert);
        indexBufferOffset += (size_t)draw_list->IdxBuffer.Size * sizeof(ImDrawIdx);
    }
}
#endif

static void ImGui_ImplMetal_DestroyTexture(ImTextureData* tex)
{
    ImGui_ImplMetal_Data* bd = ImGui_ImplMetal_GetBackendData();
    if (MetalTexture* backend_tex = (__bridge_transfer MetalTexture*)(tex->BackendUserData))
    {
        IM_ASSERT(backend_tex.metalTexture == (__bridge id<MTLTexture>)(void*)(intptr_t)tex->TexID);
#if defined(__MAC_26_0) || defined(__IPHONE_26_0)
        ImGui_ImplMetal4_ResidencySetRemoveAllocation(bd->SharedMetalContext.metal4ResidencySet, backend_tex.metalTexture);
#endif
        backend_tex.metalTexture = nil;

        // Clear identifiers and mark as destroyed (in order to allow e.g. calling InvalidateDeviceObjects while running)
        tex->SetTexID(ImTextureID_Invalid);
        tex->BackendUserData = nullptr;
    }
    tex->SetStatus(ImTextureStatus_Destroyed);
}

void ImGui_ImplMetal_UpdateTexture(ImTextureData* tex)
{
    ImGui_ImplMetal_Data* bd = ImGui_ImplMetal_GetBackendData();
    if (tex->Status == ImTextureStatus_WantCreate)
    {
        // Create and upload new texture to graphics system
        //IMGUI_DEBUG_LOG("UpdateTexture #%03d: WantCreate %dx%d\n", tex->UniqueID, tex->Width, tex->Height);
        IM_ASSERT(tex->TexID == ImTextureID_Invalid && tex->BackendUserData == nullptr);
        IM_ASSERT(tex->Format == ImTextureFormat_RGBA32);

        // We are retrieving and uploading the font atlas as a 4-channels RGBA texture here.
        // In theory we could call GetTexDataAsAlpha8() and upload a 1-channel texture to save on memory access bandwidth.
        // However, using a shader designed for 1-channel texture would make it less obvious to use the ImTextureID facility to render users own textures.
        // You can make that change in your implementation.
        MTLTextureDescriptor* textureDescriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                                     width:(NSUInteger)tex->Width
                                                                                                    height:(NSUInteger)tex->Height
                                                                                                 mipmapped:NO];
        textureDescriptor.usage = MTLTextureUsageShaderRead;
    #if TARGET_OS_OSX || TARGET_OS_MACCATALYST
        textureDescriptor.storageMode = MTLStorageModeManaged;
    #else
        textureDescriptor.storageMode = MTLStorageModeShared;
    #endif
        id <MTLTexture> texture = [bd->SharedMetalContext.device newTextureWithDescriptor:textureDescriptor];
        [texture replaceRegion:MTLRegionMake2D(0, 0, (NSUInteger)tex->Width, (NSUInteger)tex->Height) mipmapLevel:0 withBytes:tex->Pixels bytesPerRow:(NSUInteger)tex->Width * 4];
        MetalTexture* backend_tex = [[MetalTexture alloc] initWithTexture:texture];
#if defined(__MAC_26_0) || defined(__IPHONE_26_0)
        ImGui_ImplMetal4_ResidencySetAddAllocation(bd->SharedMetalContext.metal4ResidencySet, texture);
#endif

        // Store identifiers
        tex->SetTexID((ImTextureID)(intptr_t)texture);
        tex->SetStatus(ImTextureStatus_OK);
        tex->BackendUserData = (__bridge_retained void*)(backend_tex);
    }
    else if (tex->Status == ImTextureStatus_WantUpdates)
    {
        // Update selected blocks. We only ever write to textures regions which have never been used before!
        // This backend choose to use tex->Updates[] but you can use tex->UpdateRect to upload a single region.
        MetalTexture* backend_tex = (__bridge MetalTexture*)(tex->BackendUserData);
        for (ImTextureRect& r : tex->Updates)
        {
            [backend_tex.metalTexture replaceRegion:MTLRegionMake2D((NSUInteger)r.x, (NSUInteger)r.y, (NSUInteger)r.w, (NSUInteger)r.h)
                                        mipmapLevel:0
                                          withBytes:tex->GetPixelsAt(r.x, r.y)
                                        bytesPerRow:(NSUInteger)tex->Width * 4];
        }
        tex->SetStatus(ImTextureStatus_OK);
    }
    else if (tex->Status == ImTextureStatus_WantDestroy && tex->UnusedFrames > 0)
    {
        ImGui_ImplMetal_DestroyTexture(tex);
    }
}

bool ImGui_ImplMetal_CreateDeviceObjects(id<MTLDevice> device)
{
    ImGui_ImplMetal_Data* bd = ImGui_ImplMetal_GetBackendData();
    MTLDepthStencilDescriptor* depthStencilDescriptor = [[MTLDepthStencilDescriptor alloc] init];
    depthStencilDescriptor.depthWriteEnabled = NO;
    depthStencilDescriptor.depthCompareFunction = MTLCompareFunctionAlways;
    bd->SharedMetalContext.depthStencilState = [device newDepthStencilStateWithDescriptor:depthStencilDescriptor];
#ifdef IMGUI_IMPL_METAL_CPP
    [depthStencilDescriptor release];
#endif

#if defined(__MAC_26_0) || defined(__IPHONE_26_0)
    if ([device respondsToSelector:@selector(newArgumentTableWithDescriptor:error:)] && bd->SharedMetalContext.renderArgumentTableDescriptor == nil)
    {
        MTL4ArgumentTableDescriptor* argumentTableDescriptor = [[MTL4ArgumentTableDescriptor alloc] init];
        argumentTableDescriptor.maxBufferBindCount = 3;
        argumentTableDescriptor.maxTextureBindCount = 1;
        argumentTableDescriptor.supportAttributeStrides = NO;
        argumentTableDescriptor.initializeBindings = YES;
        bd->SharedMetalContext.renderArgumentTableDescriptor = argumentTableDescriptor;
    }

    if ([device respondsToSelector:@selector(newResidencySetWithDescriptor:error:)] && bd->SharedMetalContext.metal4ResidencySet == nil)
    {
        NSError* error = nil;
        MTLResidencySetDescriptor* residencyDescriptor = [[MTLResidencySetDescriptor alloc] init];
        residencyDescriptor.label = @"ImGui_ImplMetal4";
        bd->SharedMetalContext.metal4ResidencySet = [device newResidencySetWithDescriptor:residencyDescriptor error:&error];
        if (error != nil)
            NSLog(@"Error: failed to create Metal4 residency set: %@", error);
    }
#endif

    return true;
}

void ImGui_ImplMetal_DestroyDeviceObjects()
{
    ImGui_ImplMetal_Data* bd = ImGui_ImplMetal_GetBackendData();

    // Destroy all textures
    for (ImTextureData* tex : ImGui::GetPlatformIO().Textures)
        if (tex->RefCount == 1)
            ImGui_ImplMetal_DestroyTexture(tex);

    [bd->SharedMetalContext.renderPipelineStateCache removeAllObjects];
#if defined(__MAC_26_0) || defined(__IPHONE_26_0)
    bd->SharedMetalContext.renderArgumentTableDescriptor = nil;
    bd->SharedMetalContext.metal4ResidencySet = nil;
    bd->SharedMetalContext.metal4FrameResources = nil;
#endif
}

#pragma mark - MetalBuffer implementation

@implementation MetalBuffer
- (instancetype)initWithBuffer:(id<MTLBuffer>)buffer
{
    if ((self = [super init]))
    {
        _buffer = buffer;
        _lastReuseTime = GetMachAbsoluteTimeInSeconds();
    }
    return self;
}
@end

#pragma mark - FramebufferDescriptor implementation

@implementation FramebufferDescriptor
- (instancetype)initWithRenderPassDescriptor:(MTLRenderPassDescriptor*)renderPassDescriptor
{
    if ((self = [super init]))
    {
        _sampleCount = renderPassDescriptor.colorAttachments[0].texture.sampleCount;
        _colorPixelFormat = renderPassDescriptor.colorAttachments[0].texture.pixelFormat;
        _depthPixelFormat = renderPassDescriptor.depthAttachment.texture.pixelFormat;
        _stencilPixelFormat = renderPassDescriptor.stencilAttachment.texture.pixelFormat;
    }
    return self;
}

#if defined(__MAC_26_0) || defined(__IPHONE_26_0)
- (instancetype)initWithRenderPassDescriptor4:(MTL4RenderPassDescriptor*)renderPassDescriptor
{
    if ((self = [super init]))
    {
        _sampleCount = renderPassDescriptor.colorAttachments[0].texture.sampleCount;
        _colorPixelFormat = renderPassDescriptor.colorAttachments[0].texture.pixelFormat;
        _depthPixelFormat = renderPassDescriptor.depthAttachment.texture.pixelFormat;
        _stencilPixelFormat = renderPassDescriptor.stencilAttachment.texture.pixelFormat;
    }
    return self;
}
#endif

- (nonnull id)copyWithZone:(nullable NSZone*)zone
{
    FramebufferDescriptor* copy = [[FramebufferDescriptor allocWithZone:zone] init];
    copy.sampleCount = self.sampleCount;
    copy.colorPixelFormat = self.colorPixelFormat;
    copy.depthPixelFormat = self.depthPixelFormat;
    copy.stencilPixelFormat = self.stencilPixelFormat;
    return copy;
}

- (NSUInteger)hash
{
    NSUInteger sc = _sampleCount & 0x3;
    NSUInteger cf = _colorPixelFormat & 0x3FF;
    NSUInteger df = _depthPixelFormat & 0x3FF;
    NSUInteger sf = _stencilPixelFormat & 0x3FF;
    NSUInteger hash = (sf << 22) | (df << 12) | (cf << 2) | sc;
    return hash;
}

- (BOOL)isEqual:(id)object
{
    FramebufferDescriptor* other = object;
    if (![other isKindOfClass:[FramebufferDescriptor class]])
        return NO;
    return other.sampleCount == self.sampleCount      &&
    other.colorPixelFormat   == self.colorPixelFormat &&
    other.depthPixelFormat   == self.depthPixelFormat &&
    other.stencilPixelFormat == self.stencilPixelFormat;
}

@end

#pragma mark - MetalTexture implementation

@implementation MetalTexture
- (instancetype)initWithTexture:(id<MTLTexture>)metalTexture
{
    if ((self = [super init]))
        self.metalTexture = metalTexture;
    return self;
}

@end

#if defined(__MAC_26_0) || defined(__IPHONE_26_0)
@implementation MetalFrameResources
@end
#endif

#pragma mark - MetalContext implementation

@implementation MetalContext
- (instancetype)init
{
    if ((self = [super init]))
    {
        self.renderPipelineStateCache = [NSMutableDictionary dictionary];
#if defined(__MAC_26_0) || defined(__IPHONE_26_0)
        self.metal4RenderPipelineStateCache = [NSMutableDictionary dictionary];
#endif
        self.bufferCache = [NSMutableArray array];
        _lastBufferCachePurge = GetMachAbsoluteTimeInSeconds();
    }
    return self;
}

- (MetalBuffer*)dequeueReusableBufferOfLength:(NSUInteger)length device:(id<MTLDevice>)device
{
    uint64_t now = GetMachAbsoluteTimeInSeconds();

    @synchronized(self.bufferCache)
    {
        // Purge old buffers that haven't been useful for a while
        if (now - self.lastBufferCachePurge > 1.0)
        {
            NSMutableArray* survivors = [NSMutableArray array];
            for (MetalBuffer* candidate in self.bufferCache)
                if (candidate.lastReuseTime > self.lastBufferCachePurge)
                    [survivors addObject:candidate];
            self.bufferCache = [survivors mutableCopy];
            self.lastBufferCachePurge = now;
        }

        // See if we have a buffer we can reuse
        MetalBuffer* bestCandidate = nil;
        for (MetalBuffer* candidate in self.bufferCache)
            if (candidate.buffer.length >= length && (bestCandidate == nil || bestCandidate.lastReuseTime > candidate.lastReuseTime))
                bestCandidate = candidate;

        if (bestCandidate != nil)
        {
            [self.bufferCache removeObject:bestCandidate];
            bestCandidate.lastReuseTime = now;
            return bestCandidate;
        }
    }

    // No luck; make a new buffer
    id<MTLBuffer> backing = [device newBufferWithLength:length options:MTLResourceStorageModeShared];
    return [[MetalBuffer alloc] initWithBuffer:backing];
}

// Bilinear sampling is required by default. Set 'io.Fonts->Flags |= ImFontAtlasFlags_NoBakedLines' or 'style.AntiAliasedLinesUseTex = false' to allow point/nearest sampling.
- (id<MTLRenderPipelineState>)renderPipelineStateForFramebufferDescriptor:(FramebufferDescriptor*)descriptor device:(id<MTLDevice>)device
{
    NSError* error = nil;

    NSString* shaderSource = @""
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "\n"
    "struct Uniforms {\n"
    "    float4x4 projectionMatrix;\n"
    "};\n"
    "\n"
    "struct VertexIn {\n"
    "    float2 position  [[attribute(0)]];\n"
    "    float2 texCoords [[attribute(1)]];\n"
    "    uchar4 color     [[attribute(2)]];\n"
    "};\n"
    "\n"
    "struct VertexOut {\n"
    "    float4 position [[position]];\n"
    "    float2 texCoords;\n"
    "    float4 color;\n"
    "};\n"
    "\n"
    "vertex VertexOut vertex_main(VertexIn in                 [[stage_in]],\n"
    "                             constant Uniforms &uniforms [[buffer(1)]]) {\n"
    "    VertexOut out;\n"
    "    out.position = uniforms.projectionMatrix * float4(in.position, 0, 1);\n"
    "    out.texCoords = in.texCoords;\n"
    "    out.color = float4(in.color) / float4(255.0);\n"
    "    return out;\n"
    "}\n"
    "\n"
    "fragment half4 fragment_main(VertexOut in [[stage_in]],\n"
    "                             texture2d<half, access::sample> texture [[texture(0)]]) {\n"
    "    constexpr sampler linearSampler(coord::normalized, min_filter::linear, mag_filter::linear, mip_filter::linear);\n"
    "    half4 texColor = texture.sample(linearSampler, in.texCoords);\n"
    "    return half4(in.color) * texColor;\n"
    "}\n";

    id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:nil error:&error];
    if (library == nil)
    {
        NSLog(@"Error: failed to create Metal library: %@", error);
        return nil;
    }

    id<MTLFunction> vertexFunction = [library newFunctionWithName:@"vertex_main"];
    id<MTLFunction> fragmentFunction = [library newFunctionWithName:@"fragment_main"];

    if (vertexFunction == nil || fragmentFunction == nil)
    {
        NSLog(@"Error: failed to find Metal shader functions in library: %@", error);
        return nil;
    }

    MTLRenderPipelineDescriptor* pipelineDescriptor = [[MTLRenderPipelineDescriptor alloc] init];
    pipelineDescriptor.vertexFunction = vertexFunction;
    pipelineDescriptor.fragmentFunction = fragmentFunction;
    MTLVertexDescriptor* vertexDescriptor = [MTLVertexDescriptor vertexDescriptor];
    vertexDescriptor.attributes[0].offset = offsetof(ImDrawVert, pos);
    vertexDescriptor.attributes[0].format = MTLVertexFormatFloat2;
    vertexDescriptor.attributes[0].bufferIndex = 0;
    vertexDescriptor.attributes[1].offset = offsetof(ImDrawVert, uv);
    vertexDescriptor.attributes[1].format = MTLVertexFormatFloat2;
    vertexDescriptor.attributes[1].bufferIndex = 0;
    vertexDescriptor.attributes[2].offset = offsetof(ImDrawVert, col);
    vertexDescriptor.attributes[2].format = MTLVertexFormatUChar4;
    vertexDescriptor.attributes[2].bufferIndex = 0;
    vertexDescriptor.layouts[0].stepRate = 1;
    vertexDescriptor.layouts[0].stepFunction = MTLVertexStepFunctionPerVertex;
    vertexDescriptor.layouts[0].stride = sizeof(ImDrawVert);
    pipelineDescriptor.vertexDescriptor = vertexDescriptor;
    pipelineDescriptor.rasterSampleCount = self.framebufferDescriptor.sampleCount;
    pipelineDescriptor.colorAttachments[0].pixelFormat = self.framebufferDescriptor.colorPixelFormat;
    pipelineDescriptor.colorAttachments[0].blendingEnabled = YES;
    pipelineDescriptor.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
    pipelineDescriptor.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
    pipelineDescriptor.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    pipelineDescriptor.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
    pipelineDescriptor.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
    pipelineDescriptor.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    pipelineDescriptor.depthAttachmentPixelFormat = self.framebufferDescriptor.depthPixelFormat;
    pipelineDescriptor.stencilAttachmentPixelFormat = self.framebufferDescriptor.stencilPixelFormat;

    id<MTLRenderPipelineState> renderPipelineState = [device newRenderPipelineStateWithDescriptor:pipelineDescriptor error:&error];
    if (error != nil)
        NSLog(@"Error: failed to create Metal pipeline state: %@", error);

    return renderPipelineState;
}

#if defined(__MAC_26_0) || defined(__IPHONE_26_0)
- (id<MTLRenderPipelineState>)renderPipelineStateForFramebufferDescriptor4:(FramebufferDescriptor*)descriptor device:(id<MTLDevice>)device
{
    NSError* error = nil;

    NSString* shaderSource = @""
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "\n"
    "struct Uniforms {\n"
    "    float4x4 projectionMatrix;\n"
    "};\n"
    "\n"
    "struct VertexOut {\n"
    "    float4 position [[position]];\n"
    "    float2 texCoords;\n"
    "    float4 color;\n"
    "};\n"
    "\n"
    "vertex VertexOut vertex_main_metal4(uint vertex_id               [[vertex_id]],\n"
    "                                    constant uchar* vertex_bytes [[buffer(0)]],\n"
    "                                    constant Uniforms &uniforms  [[buffer(1)]],\n"
    "                                    constant ushort* indices     [[buffer(2)]]) {\n"
    "    VertexOut out;\n"
    "    uint idx = uint(indices[vertex_id]);\n"
    "    uint byte_offset = idx * 20u;\n"
    "    float2 position = float2(*((constant packed_float2*)(vertex_bytes + byte_offset + 0u)));\n"
    "    float2 texCoords = float2(*((constant packed_float2*)(vertex_bytes + byte_offset + 8u)));\n"
    "    uchar4 color = as_type<uchar4>(*((constant uint*)(vertex_bytes + byte_offset + 16u)));\n"
    "    out.position = uniforms.projectionMatrix * float4(position, 0, 1);\n"
    "    out.texCoords = texCoords;\n"
    "    out.color = float4(color) / float4(255.0);\n"
    "    return out;\n"
    "}\n"
    "\n"
    "fragment half4 fragment_main_metal4(VertexOut in [[stage_in]],\n"
    "                                    texture2d<half, access::sample> texture [[texture(0)]]) {\n"
    "    constexpr sampler linearSampler(coord::normalized, min_filter::linear, mag_filter::linear, mip_filter::linear);\n"
    "    half4 texColor = texture.sample(linearSampler, in.texCoords);\n"
    "    return half4(in.color) * texColor;\n"
    "}\n";

    id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:nil error:&error];
    if (library == nil)
    {
        NSLog(@"Error: failed to create Metal4 ImGui library: %@", error);
        return nil;
    }

    id<MTLFunction> vertexFunction = [library newFunctionWithName:@"vertex_main_metal4"];
    id<MTLFunction> fragmentFunction = [library newFunctionWithName:@"fragment_main_metal4"];
    if (vertexFunction == nil || fragmentFunction == nil)
    {
        NSLog(@"Error: failed to find Metal4 ImGui shader functions in library: %@", error);
        return nil;
    }

    MTLRenderPipelineDescriptor* pipelineDescriptor = [[MTLRenderPipelineDescriptor alloc] init];
    pipelineDescriptor.vertexFunction = vertexFunction;
    pipelineDescriptor.fragmentFunction = fragmentFunction;
    pipelineDescriptor.rasterSampleCount = descriptor.sampleCount;
    pipelineDescriptor.colorAttachments[0].pixelFormat = descriptor.colorPixelFormat;
    pipelineDescriptor.colorAttachments[0].blendingEnabled = YES;
    pipelineDescriptor.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
    pipelineDescriptor.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
    pipelineDescriptor.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    pipelineDescriptor.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
    pipelineDescriptor.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
    pipelineDescriptor.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    pipelineDescriptor.depthAttachmentPixelFormat = descriptor.depthPixelFormat;
    pipelineDescriptor.stencilAttachmentPixelFormat = descriptor.stencilPixelFormat;

    id<MTLRenderPipelineState> renderPipelineState = [device newRenderPipelineStateWithDescriptor:pipelineDescriptor error:&error];
    if (error != nil)
        NSLog(@"Error: failed to create Metal4 ImGui pipeline state: %@", error);

    return renderPipelineState;
}
#endif

@end

//-----------------------------------------------------------------------------

#endif // #ifndef IMGUI_DISABLE
