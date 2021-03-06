/*
* Copyright 2017-2019 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

#include <d3d11.h>
#include <iostream>
#include <unordered_map>
#include <memory>
#include <wrl.h>
#include "NvEncoder/NvEncoderD3D11.h"
#include "../Utils/Logger.h"
#include "../Utils/NvCodecUtils.h"
#include "../Common/AppEncUtils.h"
#include "../Common/AppEncUtilsD3D11.h"

using Microsoft::WRL::ComPtr;


simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

void Encode(char *szBgraFilePath, int nWidth, int nHeight, char *szOutFilePath, NvEncoderInitParam *pEncodeCLIOptions,
    int iGpu, bool bForceNv12)
{
    ComPtr<ID3D11Device> pDevice;
    ComPtr<ID3D11DeviceContext> pContext;
    ComPtr<IDXGIFactory1> pFactory;
    ComPtr<IDXGIAdapter> pAdapter;
    ComPtr<ID3D11Texture2D> pTexSysMem;

    ck(CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void **)pFactory.GetAddressOf()));
    ck(pFactory->EnumAdapters(iGpu, pAdapter.GetAddressOf()));
    ck(D3D11CreateDevice(pAdapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, NULL, 0,
        NULL, 0, D3D11_SDK_VERSION, pDevice.GetAddressOf(), NULL, pContext.GetAddressOf()));
    DXGI_ADAPTER_DESC adapterDesc;
    pAdapter->GetDesc(&adapterDesc);
    char szDesc[80];
    wcstombs(szDesc, adapterDesc.Description, sizeof(szDesc));
    std::cout << "GPU in use: " << szDesc << std::endl;

    D3D11_TEXTURE2D_DESC desc;
    ZeroMemory(&desc, sizeof(D3D11_TEXTURE2D_DESC));
    desc.Width = nWidth;
    desc.Height = nHeight;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_STAGING;
    desc.BindFlags = 0;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    ck(pDevice->CreateTexture2D(&desc, NULL, pTexSysMem.GetAddressOf()));

    std::unique_ptr<RGBToNV12ConverterD3D11> pConverter;
    if (bForceNv12)
    {
        pConverter.reset(new RGBToNV12ConverterD3D11(pDevice.Get(), pContext.Get(), nWidth, nHeight));
    }

    NvEncoderD3D11 enc(pDevice.Get(), nWidth, nHeight, bForceNv12 ? NV_ENC_BUFFER_FORMAT_NV12 : NV_ENC_BUFFER_FORMAT_ARGB);

    NV_ENC_INITIALIZE_PARAMS initializeParams = { NV_ENC_INITIALIZE_PARAMS_VER };
    NV_ENC_CONFIG encodeConfig = { NV_ENC_CONFIG_VER };
    initializeParams.encodeConfig = &encodeConfig;
    enc.CreateDefaultEncoderParams(&initializeParams, pEncodeCLIOptions->GetEncodeGUID(), pEncodeCLIOptions->GetPresetGUID());

    pEncodeCLIOptions->SetInitParams(&initializeParams, bForceNv12 ? NV_ENC_BUFFER_FORMAT_NV12 : NV_ENC_BUFFER_FORMAT_ARGB);

    enc.CreateEncoder(&initializeParams);

    std::ifstream fpBgra(szBgraFilePath, std::ifstream::in | std::ifstream::binary);
    if (!fpBgra)
    {
        std::ostringstream err;
        err << "Unable to open input file: " << szBgraFilePath << std::endl;
        throw std::invalid_argument(err.str());
    }

    std::ofstream fpOut(szOutFilePath, std::ios::out | std::ios::binary);
    if (!fpOut)
    {
        std::ostringstream err;
        err << "Unable to open output file: " << szOutFilePath << std::endl;
        throw std::invalid_argument(err.str());
    }

    int nSize = nWidth * nHeight * 4;
    std::unique_ptr<uint8_t[]> pHostFrame(new uint8_t[nSize]);
    int nFrame = 0;
    while (true) 
    {
        std::vector<std::vector<uint8_t>> vPacket;
        std::streamsize nRead = fpBgra.read(reinterpret_cast<char*>(pHostFrame.get()), nSize).gcount();
        if (nRead == nSize)
        {
            const NvEncInputFrame* encoderInputFrame = enc.GetNextInputFrame();
            D3D11_MAPPED_SUBRESOURCE map;
            ck(pContext->Map(pTexSysMem.Get(), D3D11CalcSubresource(0, 0, 1), D3D11_MAP_WRITE, 0, &map));
            for (int y = 0; y < nHeight; y++)
            {
                memcpy((uint8_t *)map.pData + y * map.RowPitch, pHostFrame.get() + y * nWidth * 4, nWidth * 4);
            }
            pContext->Unmap(pTexSysMem.Get(), D3D11CalcSubresource(0, 0, 1));
            if (bForceNv12)
            {
                ID3D11Texture2D *pNV12Textyure = reinterpret_cast<ID3D11Texture2D*>(encoderInputFrame->inputPtr);
                pConverter->ConvertRGBToNV12(pTexSysMem.Get(), pNV12Textyure);
            }
            else
            {
                ID3D11Texture2D *pTexBgra = reinterpret_cast<ID3D11Texture2D*>(encoderInputFrame->inputPtr);
                pContext->CopyResource(pTexBgra, pTexSysMem.Get());
            }
            enc.EncodeFrame(vPacket);
        }
        else
        {
            enc.EndEncode(vPacket);
        }
        nFrame += (int)vPacket.size();
        for (std::vector<uint8_t> &packet : vPacket)
        {
            fpOut.write(reinterpret_cast<char*>(packet.data()), packet.size());
        }
        if (nRead != nSize) {
            break;
        }
    }

    enc.DestroyEncoder();

    fpOut.close();
    fpBgra.close();

    std::cout << "Total frames encoded: " << nFrame << std::endl << "Saved in file " << szOutFilePath << std::endl;
}

/**
*  This sample application illustrates encoding of frames in ID3D11Texture2D textures.
*  There are 2 modes of operation demonstrated in this application.
*  In the default mode application reads RGB data from file and copies it to D3D11 textures
*  obtained from the encoder using NvEncoder::GetNextInputFrame() and the RGB texture is
*  submitted to NVENC for encoding. In the second case ("-nv12" option) the application converts
*  RGB textures to NV12 textures using DXVA's VideoProcessBlt API call and the NV12 texture is
*  submitted for encoding.
*/
int main(int argc, char **argv)
{
    char szInFilePath[256] = "";
    char szOutFilePath[256] = "out.h264";
    int nWidth = 0, nHeight = 0;
    try
    {
        NvEncoderInitParam encodeCLIOptions;
        int iGpu = 0;
        bool bForceNv12 = false;
        ParseCommandLine_AppEncD3D(argc, argv, szInFilePath, nWidth, nHeight, szOutFilePath, encodeCLIOptions, iGpu, bForceNv12);

        CheckInputFile(szInFilePath);
        ValidateResolution(nWidth, nHeight);

        Encode(szInFilePath, nWidth, nHeight, szOutFilePath, &encodeCLIOptions, iGpu, bForceNv12);
    }
    catch (const std::exception &ex)
    {
        std::cout << ex.what();
        exit(1);
    }
    return 0;
}
