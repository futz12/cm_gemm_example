#include <vulkan/vulkan.h>
#define NOMINMAX
#include <windows.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <cstring>
#include <cmath>

inline uint16_t floatToHalf(float f) {
    union { float f; uint32_t u; } fu;
    fu.f = f;
    uint32_t x = fu.u;
    uint32_t sign = (x >> 31) & 0x1;
    int32_t exponent = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = x & 0x7FFFFF;
    
    if (exponent <= 0) {
        if (exponent < -10) {
            return static_cast<uint16_t>(sign << 15);
        }
        mantissa |= 0x800000;
        uint32_t shift = 14 - exponent;
        uint32_t halfMantissa = mantissa >> shift;
        uint32_t roundBit = (mantissa >> (shift - 1)) & 1;
        halfMantissa += roundBit;
        return static_cast<uint16_t>((sign << 15) | halfMantissa);
    }
    if (exponent >= 31) {
        return static_cast<uint16_t>((sign << 15) | 0x7C00);
    }
    
    uint32_t halfMantissa = mantissa >> 13;
    uint32_t roundBit = (mantissa >> 12) & 1;
    halfMantissa += roundBit;
    if (halfMantissa > 0x3FF) {
        exponent++;
        halfMantissa = 0;
    }
    
    return static_cast<uint16_t>((sign << 15) | (exponent << 10) | halfMantissa);
}

#define VK_CHECK(call)                                                       \
    do {                                                                     \
        VkResult result = call;                                              \
        if (result != VK_SUCCESS) {                                          \
            std::cerr << "Vulkan error at " << __FILE__ << ":" << __LINE__   \
                      << " code=" << result << std::endl;                    \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

struct CoopMatProperties {
    uint32_t M, N, K;
    VkComponentTypeKHR AType, BType, CType, ResultType;
    VkScopeKHR scope;
};

class VulkanCMGEMM {
public:
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queueFamilyIndex = 0;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    
    VkBuffer bufferA = VK_NULL_HANDLE;
    VkBuffer bufferB = VK_NULL_HANDLE;
    VkBuffer bufferC = VK_NULL_HANDLE;
    VkDeviceMemory memoryA = VK_NULL_HANDLE;
    VkDeviceMemory memoryB = VK_NULL_HANDLE;
    VkDeviceMemory memoryC = VK_NULL_HANDLE;
    
    uint32_t subgroupSize = 32;
    CoopMatProperties selectedCM;
    bool hasKHRCoopMat = false;
    
    uint32_t M = 256;
    uint32_t N = 256;
    uint32_t K = 256;
    
    uint32_t TILE_M = 2;
    uint32_t TILE_N = 2;
    uint32_t TILE_K = 2;
    
    std::vector<float> hostA;
    std::vector<float> hostB;
    std::vector<float> hostB_reordered;
    std::vector<float> hostC;
    
    void init() {
        createInstance();
        pickPhysicalDevice();
        createDevice();
        getQueue();
        createCommandPool();
        queryCooperativeMatrixProperties();
        calculateTileSizes();
        createBuffers();
        createDescriptorSetLayout();
        createPipeline();
        createDescriptorPool();
        createDescriptorSet();
    }
    
    void cleanup() {
        if (device) {
            vkDeviceWaitIdle(device);
            vkFreeMemory(device, memoryA, nullptr);
            vkFreeMemory(device, memoryB, nullptr);
            vkFreeMemory(device, memoryC, nullptr);
            vkDestroyBuffer(device, bufferA, nullptr);
            vkDestroyBuffer(device, bufferB, nullptr);
            vkDestroyBuffer(device, bufferC, nullptr);
            vkDestroyDescriptorPool(device, descriptorPool, nullptr);
            vkDestroyPipeline(device, pipeline, nullptr);
            vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
            vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
            vkDestroyCommandPool(device, commandPool, nullptr);
            vkDestroyDevice(device, nullptr);
        }
        if (instance) {
            vkDestroyInstance(instance, nullptr);
        }
    }
    
private:
    void createInstance() {
        VkApplicationInfo appInfo = {};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Vulkan CM GEMM Optimized";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_2;
        
        std::vector<const char*> enabledLayers;
        
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());
        
        bool validationAvailable = false;
        for (const auto& layer : availableLayers) {
            if (strcmp(layer.layerName, "VK_LAYER_KHRONOS_validation") == 0) {
                validationAvailable = true;
                break;
            }
        }
        
        if (validationAvailable) {
            enabledLayers.push_back("VK_LAYER_KHRONOS_validation");
            std::cout << "Validation layers enabled" << std::endl;
        }
        
        VkInstanceCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;
        createInfo.enabledLayerCount = static_cast<uint32_t>(enabledLayers.size());
        createInfo.ppEnabledLayerNames = enabledLayers.data();
        createInfo.enabledExtensionCount = 0;
        createInfo.ppEnabledExtensionNames = nullptr;
        
        VK_CHECK(vkCreateInstance(&createInfo, nullptr, &instance));
        std::cout << "Vulkan instance created" << std::endl;
    }
    
    bool checkCooperativeMatrixSupport(VkPhysicalDevice phyDevice) {
        uint32_t extensionCount = 0;
        vkEnumerateDeviceExtensionProperties(phyDevice, nullptr, &extensionCount, nullptr);
        std::vector<VkExtensionProperties> extensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(phyDevice, nullptr, &extensionCount, extensions.data());
        
        bool hasCoopMatExt = false;
        for (const auto& ext : extensions) {
            if (strcmp(ext.extensionName, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) == 0) {
                hasCoopMatExt = true;
                break;
            }
        }
        
        if (!hasCoopMatExt) return false;
        
        VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopMatFeatures = {};
        coopMatFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
        coopMatFeatures.cooperativeMatrix = VK_TRUE;
        
        VkPhysicalDeviceVulkan12Features vulkan12Features = {};
        vulkan12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        vulkan12Features.pNext = &coopMatFeatures;
        
        VkPhysicalDeviceFeatures2 features2 = {};
        features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features2.pNext = &vulkan12Features;
        
        vkGetPhysicalDeviceFeatures2(phyDevice, &features2);
        
        return coopMatFeatures.cooperativeMatrix == VK_TRUE;
    }
    
    void pickPhysicalDevice() {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        if (deviceCount == 0) {
            std::cerr << "No Vulkan-capable GPUs found" << std::endl;
            exit(1);
        }
        
        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
        
        std::cout << "\nAvailable GPUs:" << std::endl;
        std::cout << "---------------" << std::endl;
        
        for (size_t i = 0; i < devices.size(); i++) {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(devices[i], &props);
            
            bool supportsCM = checkCooperativeMatrixSupport(devices[i]);
            
            std::cout << i << ": " << props.deviceName;
            if (supportsCM) {
                std::cout << " [Cooperative Matrix: YES]";
            } else {
                std::cout << " [Cooperative Matrix: NO]";
            }
            std::cout << std::endl;
        }
        
        for (size_t i = 0; i < devices.size(); i++) {
            if (checkCooperativeMatrixSupport(devices[i])) {
                physicalDevice = devices[i];
                break;
            }
        }
        
        if (physicalDevice == VK_NULL_HANDLE) {
            std::cerr << "\nNo GPU supports VK_KHR_cooperative_matrix extension!" << std::endl;
            exit(1);
        }
        
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(physicalDevice, &props);
        std::cout << "\nSelected GPU: " << props.deviceName << std::endl;
        
        VkPhysicalDeviceSubgroupProperties subgroupProps = {};
        subgroupProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
        
        VkPhysicalDeviceProperties2 props2 = {};
        props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        props2.pNext = &subgroupProps;
        vkGetPhysicalDeviceProperties2(physicalDevice, &props2);
        
        subgroupSize = subgroupProps.subgroupSize;
        std::cout << "Subgroup size: " << subgroupSize << std::endl;
    }
    
    void createDevice() {
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());
        
        for (uint32_t i = 0; i < queueFamilyCount; i++) {
            if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                queueFamilyIndex = i;
                break;
            }
        }
        
        float queuePriority = 1.0f;
        VkDeviceQueueCreateInfo queueCreateInfo = {};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        
        std::vector<const char*> deviceExtensions;
        deviceExtensions.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
        deviceExtensions.push_back(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
        
        VkPhysicalDeviceVulkan11Features vulkan11Features = {};
        vulkan11Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        vulkan11Features.storageBuffer16BitAccess = VK_TRUE;
        vulkan11Features.uniformAndStorageBuffer16BitAccess = VK_TRUE;
        
        VkPhysicalDeviceVulkan12Features vulkan12Features = {};
        vulkan12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        vulkan12Features.shaderFloat16 = VK_TRUE;
        vulkan12Features.pNext = &vulkan11Features;
        
        VkPhysicalDeviceSubgroupSizeControlFeatures subgroupSizeControlFeatures = {};
        subgroupSizeControlFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES;
        subgroupSizeControlFeatures.subgroupSizeControl = VK_TRUE;
        subgroupSizeControlFeatures.computeFullSubgroups = VK_TRUE;
        subgroupSizeControlFeatures.pNext = &vulkan12Features;
        
        VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopMatFeatures = {};
        coopMatFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
        coopMatFeatures.cooperativeMatrix = VK_TRUE;
        coopMatFeatures.pNext = &subgroupSizeControlFeatures;
        
        VkPhysicalDeviceFeatures2 features2 = {};
        features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features2.pNext = &coopMatFeatures;
        
        vkGetPhysicalDeviceFeatures2(physicalDevice, &features2);
        
        if (coopMatFeatures.cooperativeMatrix) {
            hasKHRCoopMat = true;
            std::cout << "VK_KHR_cooperative_matrix supported" << std::endl;
        }
        
        VkDeviceCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.pQueueCreateInfos = &queueCreateInfo;
        createInfo.queueCreateInfoCount = 1;
        createInfo.pEnabledFeatures = nullptr;
        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();
        createInfo.enabledLayerCount = 0;
        createInfo.ppEnabledLayerNames = nullptr;
        createInfo.pNext = &features2;
        
        VK_CHECK(vkCreateDevice(physicalDevice, &createInfo, nullptr, &device));
        std::cout << "Logical device created" << std::endl;
    }
    
    void getQueue() {
        vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);
    }
    
    void createCommandPool() {
        VkCommandPoolCreateInfo poolInfo = {};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = queueFamilyIndex;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        
        VK_CHECK(vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool));
    }
    
    void queryCooperativeMatrixProperties() {
        if (!hasKHRCoopMat) {
            std::cerr << "VK_KHR_cooperative_matrix not supported" << std::endl;
            exit(1);
        }
        
        PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR = 
            (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)
            vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
        
        if (!vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR) {
            std::cerr << "Could not load vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR" << std::endl;
            exit(1);
        }
        
        uint32_t propertyCount = 0;
        VK_CHECK(vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(physicalDevice, &propertyCount, nullptr));
        
        std::vector<VkCooperativeMatrixPropertiesKHR> properties(propertyCount);
        for (auto& prop : properties) {
            prop.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
            prop.pNext = nullptr;
        }
        
        VK_CHECK(vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(physicalDevice, &propertyCount, properties.data()));
        
        std::cout << "\nSupported Cooperative Matrix configurations:" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        
        bool found = false;
        for (const auto& prop : properties) {
            if (prop.scope == VK_SCOPE_SUBGROUP_KHR) {
                std::string aType, bType, cType;
                if (prop.AType == VK_COMPONENT_TYPE_FLOAT16_KHR) aType = "FP16";
                else if (prop.AType == VK_COMPONENT_TYPE_FLOAT32_KHR) aType = "FP32";
                else continue;
                
                if (prop.BType == VK_COMPONENT_TYPE_FLOAT16_KHR) bType = "FP16";
                else if (prop.BType == VK_COMPONENT_TYPE_FLOAT32_KHR) bType = "FP32";
                else continue;
                
                if (prop.CType == VK_COMPONENT_TYPE_FLOAT16_KHR) cType = "FP16";
                else if (prop.CType == VK_COMPONENT_TYPE_FLOAT32_KHR) cType = "FP32";
                else continue;
                
                std::cout << "M=" << prop.MSize << ", N=" << prop.NSize << ", K=" << prop.KSize
                          << " (A: " << aType << ", B: " << bType << ", C: " << cType << ")" << std::endl;
                
                if (!found && prop.AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                    prop.BType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                    prop.CType == VK_COMPONENT_TYPE_FLOAT32_KHR) {
                    selectedCM.M = prop.MSize;
                    selectedCM.N = prop.NSize;
                    selectedCM.K = prop.KSize;
                    selectedCM.AType = prop.AType;
                    selectedCM.BType = prop.BType;
                    selectedCM.CType = prop.CType;
                    selectedCM.ResultType = prop.ResultType;
                    selectedCM.scope = prop.scope;
                    found = true;
                }
            }
        }
        
        if (!found) {
            std::cerr << "No suitable FP16->FP32 cooperative matrix configuration found" << std::endl;
            exit(1);
        }
        
        std::cout << "\nSelected configuration: M=" << selectedCM.M << ", N=" << selectedCM.N 
                  << ", K=" << selectedCM.K << " (FP16 input, FP32 accumulate)" << std::endl;
    }
    
    void calculateTileSizes() {
        TILE_M = std::min((M + selectedCM.M - 1) / selectedCM.M, 2u);
        TILE_N = std::min((N + selectedCM.N - 1) / selectedCM.N, 2u);
        TILE_K = std::min((K + selectedCM.K - 1) / selectedCM.K, 2u);
        
        std::cout << "\nTile sizes (for double buffer optimization):" << std::endl;
        std::cout << "  TILE_M = " << TILE_M << std::endl;
        std::cout << "  TILE_N = " << TILE_N << std::endl;
        std::cout << "  TILE_K = " << TILE_K << std::endl;
    }
    
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
        
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }
        
        std::cerr << "Failed to find suitable memory type" << std::endl;
        exit(1);
    }
    
    void createBuffers() {
        uint32_t numWG_N = (N + TILE_N * selectedCM.N - 1) / (TILE_N * selectedCM.N);
        uint32_t numKTiles = (K + TILE_K * selectedCM.K - 1) / (TILE_K * selectedCM.K);
        uint32_t tileSizeB = TILE_K * selectedCM.K * TILE_N * selectedCM.N;
        size_t reorderedBSize = numWG_N * numKTiles * tileSizeB * sizeof(uint16_t);
        
        size_t sizeA = M * K * sizeof(uint16_t);
        size_t sizeB = reorderedBSize;
        size_t sizeC = M * N * sizeof(float);
        
        std::cout << "\nBuffer sizes:" << std::endl;
        std::cout << "  A: " << sizeA << " bytes (" << M << "x" << K << " FP16)" << std::endl;
        std::cout << "  B (reordered): " << sizeB << " bytes" << std::endl;
        std::cout << "  C: " << sizeC << " bytes (" << M << "x" << N << " FP32)" << std::endl;
        
        VkBufferCreateInfo bufferInfo = {};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = sizeA;
        bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        
        VK_CHECK(vkCreateBuffer(device, &bufferInfo, nullptr, &bufferA));
        
        bufferInfo.size = sizeB;
        VK_CHECK(vkCreateBuffer(device, &bufferInfo, nullptr, &bufferB));
        
        bufferInfo.size = sizeC;
        VK_CHECK(vkCreateBuffer(device, &bufferInfo, nullptr, &bufferC));
        
        VkMemoryRequirements memReqsA, memReqsB, memReqsC;
        vkGetBufferMemoryRequirements(device, bufferA, &memReqsA);
        vkGetBufferMemoryRequirements(device, bufferB, &memReqsB);
        vkGetBufferMemoryRequirements(device, bufferC, &memReqsC);
        
        VkMemoryAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memReqsA.size;
        allocInfo.memoryTypeIndex = findMemoryType(memReqsA.memoryTypeBits, 
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        
        VK_CHECK(vkAllocateMemory(device, &allocInfo, nullptr, &memoryA));
        VK_CHECK(vkBindBufferMemory(device, bufferA, memoryA, 0));
        
        allocInfo.allocationSize = memReqsB.size;
        allocInfo.memoryTypeIndex = findMemoryType(memReqsB.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VK_CHECK(vkAllocateMemory(device, &allocInfo, nullptr, &memoryB));
        VK_CHECK(vkBindBufferMemory(device, bufferB, memoryB, 0));
        
        allocInfo.allocationSize = memReqsC.size;
        allocInfo.memoryTypeIndex = findMemoryType(memReqsC.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VK_CHECK(vkAllocateMemory(device, &allocInfo, nullptr, &memoryC));
        VK_CHECK(vkBindBufferMemory(device, bufferC, memoryC, 0));
        
        std::cout << "Buffers created successfully" << std::endl;
    }
    
    void createDescriptorSetLayout() {
        VkDescriptorSetLayoutBinding bindings[3] = {};
        
        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        
        bindings[1].binding = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        
        bindings[2].binding = 2;
        bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[2].descriptorCount = 1;
        bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        
        VkDescriptorSetLayoutCreateInfo layoutInfo = {};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = 3;
        layoutInfo.pBindings = bindings;
        
        VK_CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout));
    }
    
    std::vector<uint32_t> compileShader(const std::string& source) {
        std::vector<uint32_t> spirv;
        
        char tempPath[MAX_PATH];
        GetTempPathA(MAX_PATH, tempPath);
        
        std::string tempFile = std::string(tempPath) + "cm_gemm_temp_" + std::to_string(GetCurrentProcessId()) + ".comp";
        std::string spvFile = std::string(tempPath) + "cm_gemm_temp_" + std::to_string(GetCurrentProcessId()) + ".spv";
        
        std::ofstream outFile(tempFile, std::ios::binary);
        outFile.write(source.c_str(), source.size());
        outFile.close();
        
        std::string cmd = std::string("glslangValidator -V --target-env vulkan1.2 -S comp \"") + tempFile + "\" -o \"" + spvFile + "\" 2>&1";
        
        FILE* pipe = _popen(cmd.c_str(), "r");
        if (!pipe) {
            std::cerr << "Failed to run glslangValidator" << std::endl;
            std::remove(tempFile.c_str());
            exit(1);
        }
        
        char buffer[256];
        std::string result;
        while (fgets(buffer, sizeof(buffer), pipe)) {
            result += buffer;
        }
        _pclose(pipe);
        
        std::remove(tempFile.c_str());
        
        if (result.find("error") != std::string::npos) {
            std::cerr << "Shader compilation failed:\n" << result << std::endl;
            std::remove(spvFile.c_str());
            exit(1);
        }
        
        std::ifstream spvInFile(spvFile, std::ios::binary | std::ios::ate);
        if (!spvInFile.is_open()) {
            std::cerr << "Failed to open compiled shader" << std::endl;
            std::remove(spvFile.c_str());
            exit(1);
        }
        
        size_t fileSize = spvInFile.tellg();
        spvInFile.seekg(0, std::ios::beg);
        
        spirv.resize(fileSize / sizeof(uint32_t));
        spvInFile.read(reinterpret_cast<char*>(spirv.data()), fileSize);
        spvInFile.close();
        
        std::remove(spvFile.c_str());
        
        return spirv;
    }
    
    std::string loadShaderSource() {
        std::string source = R"(#version 450
#extension GL_EXT_control_flow_attributes : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_cooperative_matrix : require

layout(constant_id = 0) const uint CM_M = 16;
layout(constant_id = 1) const uint CM_N = 16;
layout(constant_id = 2) const uint CM_K = 16;
layout(constant_id = 3) const uint subgroup_size = 32;
layout(constant_id = 4) const uint TILE_M = 2;
layout(constant_id = 5) const uint TILE_N = 2;
layout(constant_id = 6) const uint TILE_K = 2;
layout(constant_id = 7) const uint numKTiles = 1;

layout(local_size_x_id = 3) in;

layout(binding = 0) readonly buffer A_buffer { float16_t A_data[]; };
layout(binding = 1) readonly buffer B_buffer { float16_t B_data[]; };
layout(binding = 2) writeonly buffer C_buffer { float C_data[]; };

layout(push_constant) uniform PushConstants { uint M; uint N; uint K; } pc;

shared float16_t sharedA[2][TILE_M * CM_M * TILE_K * CM_K];
shared float16_t sharedB[2][TILE_K * CM_K * TILE_N * CM_N];
shared float sharedC[TILE_M * CM_M * TILE_N * CM_N];

void loadTileA(uint tileIdx, uint wgRow, uint kStart) {
    const uint lane = gl_SubgroupInvocationID;
    const uint totalElements = TILE_M * CM_M * TILE_K * CM_K;
    for (uint i = lane; i < totalElements; i += subgroup_size) {
        uint localM = i / (TILE_K * CM_K);
        uint localK = i % (TILE_K * CM_K);
        uint globalM = wgRow * TILE_M * CM_M + localM;
        uint globalK = kStart + localK;
        if (globalM < pc.M && globalK < pc.K) {
            sharedA[tileIdx][i] = A_data[globalM * pc.K + globalK];
        } else {
            sharedA[tileIdx][i] = float16_t(0.0);
        }
    }
}

void loadTileB(uint tileIdx, uint wgCol, uint kt) {
    const uint lane = gl_SubgroupInvocationID;
    const uint tileSizeB = TILE_K * CM_K * TILE_N * CM_N;
    uint tileOffset = (wgCol * numKTiles + kt) * tileSizeB;
    for (uint i = lane; i < tileSizeB; i += subgroup_size) {
        sharedB[tileIdx][i] = B_data[tileOffset + i];
    }
}

void main() {
    const uint wgRow = gl_WorkGroupID.x;
    const uint wgCol = gl_WorkGroupID.y;
    const uint lane = gl_SubgroupInvocationID;

    if (wgRow * TILE_M * CM_M >= pc.M || wgCol * TILE_N * CM_N >= pc.N) return;

    coopmat<float, gl_ScopeSubgroup, CM_M, CM_N, gl_MatrixUseAccumulator> sum[TILE_M][TILE_N];
    [[unroll]] for (uint tm = 0; tm < TILE_M; tm++) {
        [[unroll]] for (uint tn = 0; tn < TILE_N; tn++) {
            sum[tm][tn] = coopmat<float, gl_ScopeSubgroup, CM_M, CM_N, gl_MatrixUseAccumulator>(0.f);
        }
    }

    const uint KTileSize = TILE_K * CM_K;

    loadTileA(0, wgRow, 0);
    loadTileB(0, wgCol, 0);
    barrier();

    for (uint kt = 0; kt < numKTiles; kt++) {
        uint currentBuf = kt % 2;
        uint nextBuf = (kt + 1) % 2;

        [[unroll]] for (uint tk = 0; tk < TILE_K; tk++) {
            [[unroll]] for (uint tm = 0; tm < TILE_M; tm++) {
                [[unroll]] for (uint tn = 0; tn < TILE_N; tn++) {
                    coopmat<float16_t, gl_ScopeSubgroup, CM_M, CM_K, gl_MatrixUseA> matA;
                    coopmat<float16_t, gl_ScopeSubgroup, CM_K, CM_N, gl_MatrixUseB> matB;

                    uint offsetA = (tm * CM_M) * (TILE_K * CM_K) + tk * CM_K;
                    uint offsetB = (tk * CM_K) * (TILE_N * CM_N) + tn * CM_N;
                    uint strideA = TILE_K * CM_K;
                    uint strideB = TILE_N * CM_N;

                    coopMatLoad(matA, sharedA[currentBuf], offsetA, strideA, gl_CooperativeMatrixLayoutRowMajor);
                    coopMatLoad(matB, sharedB[currentBuf], offsetB, strideB, gl_CooperativeMatrixLayoutRowMajor);

                    sum[tm][tn] = coopMatMulAdd(matA, matB, sum[tm][tn]);
                }
            }
        }

        barrier();

        uint nextKt = kt + 1;
        if (nextKt < numKTiles) {
            uint nextKStart = nextKt * KTileSize;
            loadTileA(nextBuf, wgRow, nextKStart);
            loadTileB(nextBuf, wgCol, nextKt);
        }

        barrier();
    }

    [[unroll]] for (uint tm = 0; tm < TILE_M; tm++) {
        [[unroll]] for (uint tn = 0; tn < TILE_N; tn++) {
            uint offsetC = (tm * CM_M) * (TILE_N * CM_N) + tn * CM_N;
            uint strideC = TILE_N * CM_N;
            coopMatStore(sum[tm][tn], sharedC, offsetC, strideC, gl_CooperativeMatrixLayoutRowMajor);
        }
    }
    barrier();

    const uint totalC = TILE_M * CM_M * TILE_N * CM_N;
    for (uint i = lane; i < totalC; i += subgroup_size) {
        uint localM = i / (TILE_N * CM_N);
        uint localN = i % (TILE_N * CM_N);
        uint globalM = wgRow * TILE_M * CM_M + localM;
        uint globalN = wgCol * TILE_N * CM_N + localN;
        if (globalM < pc.M && globalN < pc.N) {
            C_data[globalM * pc.N + globalN] = sharedC[i];
        }
    }
}
)";
        return source;
    }
    
    void createPipeline() {
        std::string shaderSource = loadShaderSource();
        std::vector<uint32_t> spirv = compileShader(shaderSource);
        
        VkShaderModuleCreateInfo shaderInfo = {};
        shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        shaderInfo.codeSize = spirv.size() * sizeof(uint32_t);
        shaderInfo.pCode = spirv.data();
        
        VkShaderModule shaderModule;
        VK_CHECK(vkCreateShaderModule(device, &shaderInfo, nullptr, &shaderModule));
        
        VkPushConstantRange pushConstantRange = {};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = 3 * sizeof(uint32_t);
        
        VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
        
        VK_CHECK(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout));
        
        uint32_t numKTiles = (K + TILE_K * selectedCM.K - 1) / (TILE_K * selectedCM.K);
        uint32_t specData[8] = {
            selectedCM.M, selectedCM.N, selectedCM.K,
            subgroupSize,
            TILE_M, TILE_N, TILE_K,
            numKTiles
        };
        
        VkSpecializationMapEntry mapEntries[8] = {};
        for (int i = 0; i < 8; i++) {
            mapEntries[i].constantID = i;
            mapEntries[i].offset = i * sizeof(uint32_t);
            mapEntries[i].size = sizeof(uint32_t);
        }
        
        VkSpecializationInfo specInfo = {};
        specInfo.mapEntryCount = 8;
        specInfo.pMapEntries = mapEntries;
        specInfo.dataSize = sizeof(specData);
        specInfo.pData = specData;
        
        VkComputePipelineCreateInfo pipelineInfo = {};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineInfo.stage.module = shaderModule;
        pipelineInfo.stage.pName = "main";
        pipelineInfo.stage.pSpecializationInfo = &specInfo;
        pipelineInfo.stage.flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
        pipelineInfo.layout = pipelineLayout;
        
        VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline));
        
        vkDestroyShaderModule(device, shaderModule, nullptr);
        std::cout << "Pipeline created" << std::endl;
    }
    
    void createDescriptorPool() {
        VkDescriptorPoolSize poolSize = {};
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSize.descriptorCount = 3;
        
        VkDescriptorPoolCreateInfo poolInfo = {};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;
        
        VK_CHECK(vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool));
    }
    
    void createDescriptorSet() {
        VkDescriptorSetAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &descriptorSetLayout;
        
        VK_CHECK(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet));
        
        VkDescriptorBufferInfo bufferInfos[3] = {};
        bufferInfos[0].buffer = bufferA;
        bufferInfos[0].offset = 0;
        bufferInfos[0].range = VK_WHOLE_SIZE;
        
        bufferInfos[1].buffer = bufferB;
        bufferInfos[1].offset = 0;
        bufferInfos[1].range = VK_WHOLE_SIZE;
        
        bufferInfos[2].buffer = bufferC;
        bufferInfos[2].offset = 0;
        bufferInfos[2].range = VK_WHOLE_SIZE;
        
        VkWriteDescriptorSet descriptorWrites[3] = {};
        for (int i = 0; i < 3; i++) {
            descriptorWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[i].dstSet = descriptorSet;
            descriptorWrites[i].dstBinding = i;
            descriptorWrites[i].descriptorCount = 1;
            descriptorWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[i].pBufferInfo = &bufferInfos[i];
        }
        
        vkUpdateDescriptorSets(device, 3, descriptorWrites, 0, nullptr);
    }
    
public:
    void reorderBWeights() {
        std::cout << "\n=== B Weight Reordering ===" << std::endl;
        
        uint32_t numWG_N = (N + TILE_N * selectedCM.N - 1) / (TILE_N * selectedCM.N);
        uint32_t numKTiles = (K + TILE_K * selectedCM.K - 1) / (TILE_K * selectedCM.K);
        uint32_t tileSizeB = TILE_K * selectedCM.K * TILE_N * selectedCM.N;
        
        hostB_reordered.clear();
        hostB_reordered.resize(numWG_N * numKTiles * tileSizeB, 0.0f);
        
        for (uint32_t wgCol = 0; wgCol < numWG_N; wgCol++) {
            for (uint32_t kt = 0; kt < numKTiles; kt++) {
                uint32_t tileOffset = (wgCol * numKTiles + kt) * tileSizeB;
                
                for (uint32_t localK = 0; localK < TILE_K * selectedCM.K; localK++) {
                    for (uint32_t localN = 0; localN < TILE_N * selectedCM.N; localN++) {
                        uint32_t globalK = kt * TILE_K * selectedCM.K + localK;
                        uint32_t globalN = wgCol * TILE_N * selectedCM.N + localN;
                        
                        uint32_t reorderedIdx = tileOffset + localK * (TILE_N * selectedCM.N) + localN;
                        
                        if (globalK < K && globalN < N) {
                            hostB_reordered[reorderedIdx] = hostB[globalK * N + globalN];
                        } else {
                            hostB_reordered[reorderedIdx] = 0.0f;
                        }
                    }
                }
            }
        }
        
        std::cout << "B weight reordered for optimal memory access pattern" << std::endl;
        std::cout << "  Original size: " << K * N << " elements" << std::endl;
        std::cout << "  Reordered size: " << hostB_reordered.size() << " elements" << std::endl;
        std::cout << "  numWG_N: " << numWG_N << ", numKTiles: " << numKTiles << std::endl;
    }
    
    void initTestData() {
        std::cout << "\n=== Initializing Test Data ===" << std::endl;
        hostA.resize(M * K);
        hostB.resize(K * N);
        hostC.resize(M * N);
        
        for (uint32_t i = 0; i < M * K; i++) {
            hostA[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
        }
        for (uint32_t i = 0; i < K * N; i++) {
            hostB[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
        }
        
        reorderBWeights();
        
        std::vector<uint16_t> hostA_fp16(M * K);
        std::vector<uint16_t> hostB_reordered_fp16(hostB_reordered.size());
        
        for (uint32_t i = 0; i < M * K; i++) {
            hostA_fp16[i] = floatToHalf(hostA[i]);
        }
        for (size_t i = 0; i < hostB_reordered.size(); i++) {
            hostB_reordered_fp16[i] = floatToHalf(hostB_reordered[i]);
        }
        
        void* pData = nullptr;
        VkResult res = vkMapMemory(device, memoryA, 0, M * K * sizeof(uint16_t), 0, &pData);
        if (res != VK_SUCCESS || pData == nullptr) {
            std::cerr << "Failed to map memory A" << std::endl;
            return;
        }
        memcpy(pData, hostA_fp16.data(), M * K * sizeof(uint16_t));
        vkUnmapMemory(device, memoryA);
        
        pData = nullptr;
        res = vkMapMemory(device, memoryB, 0, hostB_reordered_fp16.size() * sizeof(uint16_t), 0, &pData);
        if (res != VK_SUCCESS || pData == nullptr) {
            std::cerr << "Failed to map memory B" << std::endl;
            return;
        }
        memcpy(pData, hostB_reordered_fp16.data(), hostB_reordered_fp16.size() * sizeof(uint16_t));
        vkUnmapMemory(device, memoryB);
        
        std::cout << "Test data uploaded to GPU" << std::endl;
    }
    
    void runGPU() {
        VkCommandBufferAllocateInfo cmdBufInfo = {};
        cmdBufInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmdBufInfo.commandPool = commandPool;
        cmdBufInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmdBufInfo.commandBufferCount = 1;
        
        VkCommandBuffer commandBuffer;
        VK_CHECK(vkAllocateCommandBuffers(device, &cmdBufInfo, &commandBuffer));
        
        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        
        VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));
        
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
        
        uint32_t pushConstants[3] = {M, N, K};
        vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), pushConstants);
        
        uint32_t wgX = (M + TILE_M * selectedCM.M - 1) / (TILE_M * selectedCM.M);
        uint32_t wgY = (N + TILE_N * selectedCM.N - 1) / (TILE_N * selectedCM.N);
        
        std::cout << "\nDispatching " << wgX << " x " << wgY << " workgroups" << std::endl;
        
        vkCmdDispatch(commandBuffer, wgX, wgY, 1);
        
        VkMemoryBarrier barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                             0, 1, &barrier, 0, nullptr, 0, nullptr);
        
        VK_CHECK(vkEndCommandBuffer(commandBuffer));
        
        VkSubmitInfo submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
        VK_CHECK(vkQueueWaitIdle(queue));
        
        auto end = std::chrono::high_resolution_clock::now();
        auto gpuTime = std::chrono::duration<double, std::milli>(end - start).count();
        
        void* pData = nullptr;
        vkMapMemory(device, memoryC, 0, M * N * sizeof(float), 0, &pData);
        memcpy(hostC.data(), pData, M * N * sizeof(float));
        vkUnmapMemory(device, memoryC);
        
        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
        
        std::cout << "\n=== GPU Results ===" << std::endl;
        std::cout << "Execution time: " << std::fixed << std::setprecision(3) << gpuTime << " ms" << std::endl;
        
        double gflops = (2.0 * M * N * K) / (gpuTime * 1e6);
        std::cout << "Performance: " << std::fixed << std::setprecision(2) << gflops << " GFLOPS" << std::endl;
    }
};

void cpuGEMM(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C,
             uint32_t M, uint32_t N, uint32_t K) {
    for (uint32_t i = 0; i < M; i++) {
        for (uint32_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

std::vector<float> cpuGEMMTimed(const std::vector<float>& A, const std::vector<float>& B,
                                 uint32_t M, uint32_t N, uint32_t K) {
    std::vector<float> C(M * N);
    
    int warmupRuns = 3;
    for (int i = 0; i < warmupRuns; i++) {
        cpuGEMM(A, B, C, M, N, K);
    }
    
    int numRuns = 5;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < numRuns; i++) {
        cpuGEMM(A, B, C, M, N, K);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto cpuTime = std::chrono::duration<double, std::milli>(end - start).count() / numRuns;
    
    std::cout << "\n=== CPU Results ===" << std::endl;
    std::cout << "Execution time (avg of " << numRuns << " runs): " 
              << std::fixed << std::setprecision(3) << cpuTime << " ms" << std::endl;
    
    double gflops = (2.0 * M * N * K) / (cpuTime * 1e6);
    std::cout << "Performance: " << std::fixed << std::setprecision(2) << gflops << " GFLOPS" << std::endl;
    
    return C;
}

bool verifyResults(const std::vector<float>& gpuResult, const std::vector<float>& cpuResult,
                   uint32_t M, uint32_t N, float tolerance = 1e-2f) {
    float maxDiff = 0.0f;
    float avgDiff = 0.0f;
    int errorCount = 0;
    
    for (uint32_t i = 0; i < M * N; i++) {
        float diff = std::abs(gpuResult[i] - cpuResult[i]);
        maxDiff = std::max(maxDiff, diff);
        avgDiff += diff;
        
        if (diff > tolerance) {
            errorCount++;
        }
    }
    avgDiff /= (M * N);
    
    std::cout << "\n=== Result Verification ===" << std::endl;
    std::cout << "Max difference: " << std::scientific << maxDiff << std::endl;
    std::cout << "Avg difference: " << std::scientific << avgDiff << std::endl;
    std::cout << "Errors (diff > " << std::scientific << tolerance << "): " << errorCount << "/" << M * N << std::endl;
    
    if (errorCount == 0) {
        std::cout << "Status: PASSED" << std::endl;
        return true;
    } else {
        std::cout << "Status: FAILED (tolerance exceeded)" << std::endl;
        return false;
    }
}

int main() {
    std::cout << "============================================" << std::endl;
    std::cout << "Vulkan Cooperative Matrix GEMM - Optimized" << std::endl;
    std::cout << "With Double Buffer Optimization" << std::endl;
    std::cout << "============================================" << std::endl;
    
    VulkanCMGEMM app;
    
    try {
        app.init();
        app.initTestData();
        
        std::cout << "\n=== Running GPU GEMM ===" << std::endl;
        app.runGPU();
        
        std::cout << "\n=== Running CPU GEMM for comparison ===" << std::endl;
        std::vector<float> cpuResult = cpuGEMMTimed(app.hostA, app.hostB, app.M, app.N, app.K);
        
        verifyResults(app.hostC, cpuResult, app.M, app.N);
        
        app.cleanup();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        app.cleanup();
        return 1;
    }
    
    std::cout << "\nDone!" << std::endl;
    return 0;
}
