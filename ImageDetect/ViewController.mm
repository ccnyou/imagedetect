//
//  ViewController.m
//  ImageDetect
//
//  Created by 聪宁陈 on 2018/7/20.
//  Copyright © 2018年 ccnyou. All rights reserved.
//

#import "ViewController.h"

#include <sys/time.h>
#include <fstream>
#include <iostream>
#include <queue>

#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/string_util.h"
#include "tensorflow/contrib/lite/tools/mutable_op_resolver.h"

#import "UIImage+Utils.h"

@interface ViewController ()
@end

@implementation ViewController {
    std::vector<std::string> _labels;
    tflite::ops::builtin::BuiltinOpResolver _resolver;
    std::unique_ptr<tflite::Interpreter> _interpreter;
    std::unique_ptr<tflite::FlatBufferModel> _model;
    int _total_count;
    double _total_latency;
}

#define LOG(level) std::cout

static const int wanted_input_width = 299;
static const int wanted_input_height = 299;
static const int wanted_input_channels = 3;

static NSString* FilePathForResourceName(NSString* name, NSString* extension) {
    NSString* file_path = [[NSBundle mainBundle] pathForResource:name ofType:extension];
    if (file_path == NULL) {
        NSLog(@"%s %d error", __FUNCTION__, __LINE__);
    }
    return file_path;
}

static void LoadLabels(NSString* file_name, NSString* file_type,
                       std::vector<std::string>* label_strings) {
    NSString* labels_path = FilePathForResourceName(file_name, file_type);
    if (!labels_path) {
        NSLog(@"%s %d error", __FUNCTION__, __LINE__);
    }
    std::ifstream t;
    t.open([labels_path UTF8String]);
    std::string line;
    while (t) {
        std::getline(t, line);
        label_strings->push_back(line);
    }
    t.close();
}

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    
    static NSString* model_file_name = @"trained";
    static NSString* labels_file_name = @"trained_labels";
    
    NSString* graph_path = FilePathForResourceName(model_file_name, @"tflite");
    _model = tflite::FlatBufferModel::BuildFromFile([graph_path UTF8String]);
    
    tflite::ops::builtin::BuiltinOpResolver resolver;
    LoadLabels(labels_file_name, @"txt", &_labels);
    
    tflite::InterpreterBuilder(*_model, resolver)(&_interpreter);
    if (!_interpreter) {
        NSLog(@"%s %d error", __FUNCTION__, __LINE__);
    }
    if (_interpreter->AllocateTensors() != kTfLiteOk) {
        NSLog(@"%s %d error", __FUNCTION__, __LINE__);
    }
    
    UIImage *image = [UIImage imageNamed:@"2"];
    image = [image scaleToSize:CGSizeMake(wanted_input_width, wanted_input_height)];
    image = [image cropToSize:CGSizeMake(wanted_input_width, wanted_input_height)];
    CVPixelBufferRef bufferRef = [self pixelBufferFromCGImage:image.CGImage];
    [self runModelOnFrame:bufferRef];
}


- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

static void GetTopN(const float* prediction, const int prediction_size, const int num_results,
                    const float threshold, std::vector<std::pair<float, int>>* top_results) {
    // Will contain top N results in ascending order.
    std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>,
    std::greater<std::pair<float, int>>>
    top_result_pq;
    
    const long count = prediction_size;
    for (int i = 0; i < count; ++i) {
        const float value = prediction[i]; //prediction[i] / 255.0;
        // Only add it if it beats the threshold and has a chance at being in
        // the top N.
        if (value < threshold) {
            continue;
        }
        
        top_result_pq.push(std::pair<float, int>(value, i));
        
        // If at capacity, kick the smallest value out.
        if (top_result_pq.size() > num_results) {
            top_result_pq.pop();
        }
    }
    
    // Copy to output vector and reverse into descending order.
    while (!top_result_pq.empty()) {
        top_results->push_back(top_result_pq.top());
        top_result_pq.pop();
    }
    std::reverse(top_results->begin(), top_results->end());
}

- (CVPixelBufferRef)pixelBufferFromCGImage:(CGImageRef)image
{
    
    CGSize frameSize = CGSizeMake(CGImageGetWidth(image), CGImageGetHeight(image));
    NSDictionary *options = [NSDictionary dictionaryWithObjectsAndKeys:
                             [NSNumber numberWithBool:NO], kCVPixelBufferCGImageCompatibilityKey,
                             [NSNumber numberWithBool:NO], kCVPixelBufferCGBitmapContextCompatibilityKey,
                             nil];
    CVPixelBufferRef pxbuffer = NULL;
    CVReturn status = CVPixelBufferCreate(kCFAllocatorDefault, frameSize.width,
                                          frameSize.height,  kCVPixelFormatType_32ARGB, (__bridge CFDictionaryRef) options,
                                          &pxbuffer);
    NSParameterAssert(status == kCVReturnSuccess && pxbuffer != NULL);
    
    CVPixelBufferLockBaseAddress(pxbuffer, 0);
    void *pxdata = CVPixelBufferGetBaseAddress(pxbuffer);
    
    
    CGColorSpaceRef rgbColorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef context = CGBitmapContextCreate(pxdata, frameSize.width,
                                                 frameSize.height, 8, 4*frameSize.width, rgbColorSpace,
                                                 kCGImageAlphaNoneSkipLast);
    
    CGContextDrawImage(context, CGRectMake(0, 0, CGImageGetWidth(image),
                                           CGImageGetHeight(image)), image);
    CGColorSpaceRelease(rgbColorSpace);
    CGContextRelease(context);
    
    CVPixelBufferUnlockBaseAddress(pxbuffer, 0);
    
    return pxbuffer;
}

- (void)runModelOnFrame:(CVPixelBufferRef)pixelBuffer {
    assert(pixelBuffer != NULL);
    
    OSType sourcePixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer);
    assert(sourcePixelFormat == kCVPixelFormatType_32ARGB ||
           sourcePixelFormat == kCVPixelFormatType_32BGRA);
    
    const int sourceRowBytes = (int)CVPixelBufferGetBytesPerRow(pixelBuffer);
    const int image_width = (int)CVPixelBufferGetWidth(pixelBuffer);
    const int fullHeight = (int)CVPixelBufferGetHeight(pixelBuffer);
    
    CVPixelBufferLockFlags unlockFlags = kNilOptions;
    CVPixelBufferLockBaseAddress(pixelBuffer, unlockFlags);
    
    unsigned char* sourceBaseAddr = (unsigned char*)(CVPixelBufferGetBaseAddress(pixelBuffer));
    int image_height;
    unsigned char* sourceStartAddr;
    if (fullHeight <= image_width) {
        image_height = fullHeight;
        sourceStartAddr = sourceBaseAddr;
    } else {
        image_height = image_width;
        const int marginY = ((fullHeight - image_width) / 2);
        sourceStartAddr = (sourceBaseAddr + (marginY * sourceRowBytes));
    }
    const int image_channels = 4;
    assert(image_channels >= wanted_input_channels);
    uint8_t* in = sourceStartAddr;
    
    const auto &inputs = _interpreter->inputs();
    int input = inputs[0];
    
    float* out = _interpreter->typed_tensor<float>(input);
    for (int y = 0; y < wanted_input_height; ++y) {
        float* out_row = out + (y * wanted_input_width * wanted_input_channels);
        for (int x = 0; x < wanted_input_width; ++x) {
            const int in_x = (y * image_width) / wanted_input_width;
            const int in_y = (x * image_height) / wanted_input_height;
            uint8_t* in_pixel = in + (in_y * image_width * image_channels) + (in_x * image_channels);
            float* out_pixel = out_row + (x * wanted_input_channels);
            for (int c = 0; c < wanted_input_channels; ++c) {
                out_pixel[c] = in_pixel[c];
            }
        }
    }
    
    double startTimestamp = [[NSDate new] timeIntervalSince1970];
    if (_interpreter->Invoke() != kTfLiteOk) {
        LOG(FATAL) << "Failed to invoke!";
    }
    double endTimestamp = [[NSDate new] timeIntervalSince1970];
    _total_latency += (endTimestamp - startTimestamp);
    _total_count += 1;
    NSLog(@"Time: %.4lf, avg: %.4lf, count: %d", endTimestamp - startTimestamp,
          _total_latency / _total_count, _total_count);
    
    const int output_size = 7;
    const int kNumResults = 5;
    const float kThreshold = 0.1f;
    
    std::vector<std::pair<float, int>> top_results;
    
    float* output = _interpreter->typed_output_tensor<float>(0);
    GetTopN(output, output_size, kNumResults, kThreshold, &top_results);
    
    NSMutableDictionary* newValues = [NSMutableDictionary dictionary];
    for (const auto& result : top_results) {
        const float confidence = result.first;
        const int index = result.second;
        NSString* labelObject = [NSString stringWithUTF8String:_labels[index].c_str()];
        if (!labelObject) {
            continue;
        }
        NSNumber* valueObject = [NSNumber numberWithFloat:confidence];
        [newValues setObject:valueObject forKey:labelObject];
    }
    dispatch_async(dispatch_get_main_queue(), ^(void) {
        [self setPredictionValues:newValues];
    });
    
    CVPixelBufferUnlockBaseAddress(pixelBuffer, unlockFlags);
    CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
}

- (void)setPredictionValues:(NSMutableDictionary *)values {
    NSLog(@"%s %d values = %@", __FUNCTION__, __LINE__, values);
}

@end
