// entry/src/main/ets/common/Haptics.ets
import vibrator from '@ohos.vibrator'
import { hilog } from '@kit.PerformanceAnalysisKit'
import { BusinessError } from '@ohos.base'

const DOMAIN = 0x0000
const TAG: string = 'HAPTIC'

export class Haptics {
  static async vibrateShort(): Promise<void> {
    // 日志 1：明确“振动方法被调用”
    hilog.info(DOMAIN, TAG, 'vibrateShort() called')

    try {
      await new Promise<void>((resolve) => {
        vibrator.startVibration(
          { type: 'time', duration: 120 },
          { id: 0, usage: 'unknown' },
          (error: BusinessError) => {
            if (error) {
              // 日志 2：振动 API 返回错误（模拟器一般不会报错）
              hilog.error(DOMAIN, TAG,
                'startVibration callback error: %{public}s',
                JSON.stringify(error)
              )
            } else {
              // 日志 3：振动 API 已正常执行
              hilog.info(DOMAIN, TAG, 'startVibration callback ok')
            }
            resolve()
          }
        )
      })
    } catch (e) {
      // 日志 4：极端异常兜底
      hilog.error(DOMAIN, TAG,
        'vibrateShort exception: %{public}s',
        JSON.stringify(e)
      )
    }
  }
}
