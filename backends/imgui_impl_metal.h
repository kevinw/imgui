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

#pragma once
#include "imgui.h"      // IMGUI_IMPL_API
#ifndef IMGUI_DISABLE

//-----------------------------------------------------------------------------
// ObjC API
//-----------------------------------------------------------------------------

#ifdef __OBJC__

@class MTLRenderPassDescriptor;
@class MTL4RenderPassDescriptor;
@protocol MTLDevice, MTLCommandBuffer, MTLRenderCommandEncoder;
@protocol MTLSharedEvent;
@protocol MTL4CommandBuffer, MTL4RenderCommandEncoder, MTL4CommandQueue;

// Follow "Getting Started" link and check examples/ folder to learn about using backends!
IMGUI_IMPL_API bool ImGui_ImplMetal_Init(id<MTLDevice> device);
IMGUI_IMPL_API void ImGui_ImplMetal_Shutdown();
IMGUI_IMPL_API void ImGui_ImplMetal_NewFrame(MTLRenderPassDescriptor* renderPassDescriptor);
IMGUI_IMPL_API void ImGui_ImplMetal_RenderDrawData(ImDrawData* drawData,
                                                   id<MTLCommandBuffer> commandBuffer,
                                                   id<MTLRenderCommandEncoder> commandEncoder);
#if defined(__MAC_26_0) || defined(__IPHONE_26_0)
IMGUI_IMPL_API void ImGui_ImplMetal4_NewFrame(MTL4RenderPassDescriptor* renderPassDescriptor);
IMGUI_IMPL_API void ImGui_ImplMetal4_RenderDrawData(ImDrawData* drawData,
                                                    id<MTL4CommandBuffer> commandBuffer,
                                                    id<MTL4RenderCommandEncoder> commandEncoder);
IMGUI_IMPL_API void ImGui_ImplMetal4_ConfigureFrameSynchronization(id<MTLSharedEvent> sharedEvent, int maxFramesInFlight);
IMGUI_IMPL_API void ImGui_ImplMetal4_NotifyFrameSubmitted(id<MTL4CommandQueue> commandQueue);
#endif

// Called by Init/NewFrame/Shutdown
IMGUI_IMPL_API bool ImGui_ImplMetal_CreateDeviceObjects(id<MTLDevice> device);
IMGUI_IMPL_API void ImGui_ImplMetal_DestroyDeviceObjects();

// (Advanced) Use e.g. if you need to precisely control the timing of texture updates (e.g. for staged rendering), by setting ImDrawData::Textures = nullptr to handle this manually.
IMGUI_IMPL_API void ImGui_ImplMetal_UpdateTexture(ImTextureData* tex);

#endif

//-----------------------------------------------------------------------------
// C++ API
//-----------------------------------------------------------------------------

// Enable Metal C++ binding support with '#define IMGUI_IMPL_METAL_CPP' in your imconfig.h file
// More info about using Metal from C++: https://developer.apple.com/metal/cpp/

#ifdef IMGUI_IMPL_METAL_CPP
#include <Metal/Metal.hpp>
#ifndef __OBJC__

// Follow "Getting Started" link and check examples/ folder to learn about using backends!
IMGUI_IMPL_API bool ImGui_ImplMetal_Init(MTL::Device* device);
IMGUI_IMPL_API void ImGui_ImplMetal_Shutdown();
IMGUI_IMPL_API void ImGui_ImplMetal_NewFrame(MTL::RenderPassDescriptor* renderPassDescriptor);
IMGUI_IMPL_API void ImGui_ImplMetal_RenderDrawData(ImDrawData* draw_data,
                                                   MTL::CommandBuffer* commandBuffer,
                                                   MTL::RenderCommandEncoder* commandEncoder);

// Called by Init/NewFrame/Shutdown
IMGUI_IMPL_API bool ImGui_ImplMetal_CreateDeviceObjects(MTL::Device* device);
IMGUI_IMPL_API void ImGui_ImplMetal_DestroyDeviceObjects();

// (Advanced) Use e.g. if you need to precisely control the timing of texture updates (e.g. for staged rendering), by setting ImDrawData::Textures = nullptr to handle this manually.
IMGUI_IMPL_API void ImGui_ImplMetal_UpdateTexture(ImTextureData* tex);

#endif
#endif

//-----------------------------------------------------------------------------

#endif // #ifndef IMGUI_DISABLE
