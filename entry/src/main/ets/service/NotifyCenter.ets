import notificationManager from '@ohos.notificationManager'
import notification from '@ohos.notification'
import { hilog } from '@kit.PerformanceAnalysisKit'
import { Haptics } from '../common/Haptics'
import wantAgent from '@ohos.wantAgent'  // 用 WantAgent 绑定“点击通知后跳转”的动作（ArkTS 6 可用）

// hilog 的 domain/tag：用于日志分类过滤（domain 常量一般随模块固定，tag 用模块名）
const DOMAIN: number = 0x0000
const TAG: string = 'HarmonyStock.Notify'

// 通知触发时需要的最小信息：通知文案 + 点击后要打开哪个 Ability
export interface QuoteAlertPayload {
  id: number                 // 通知 id（同 id 会覆盖更新）
  title: string              // 通知标题
  text: string               // 主文案（行情命中描述）
  additionalText: string     // 次要文案（例如“打开应用查看详情”）
  bundleName: string         // 目标应用包名（一般就是本应用包名）
  abilityName: string        // 目标 Ability（例如 EntryAbility）
}

// 统一错误结构：这里只用 code 做日志打印，避免 any/unknown
interface ErrorObject {
  code: string
}

// WantAgent wants 里每一项描述“要打开哪个应用的哪个 Ability”
interface WantAgentRequest {
  bundleName: string
  abilityName: string
}

// WantAgent 的配置：给 notificationManager.publish 用的
interface WantAgentConfig {
  wants: Array<WantAgentRequest>         // 这里必须是数组：WantAgent API 设计如此
  operationType: wantAgent.OperationType // START_ABILITY 等
  requestCode: number                    // 用通知 id 当 requestCode，便于一一对应
  wantAgentFlags: Array<wantAgent.WantAgentFlags> // UPDATE_PRESENT_FLAG 等
}

export class NotifyCenter {
  // 确保系统通知权限已开启；未开启时会拉起系统授权页面
  static async ensureEnabled(): Promise<boolean> {
    try {
      // 先直接检查当前通知是否可用
      const enabled: boolean = await notificationManager.isNotificationEnabled()
      if (enabled) {
        return true
      }

      // 未开启则请求系统弹出“开启通知”的设置界面
      await notificationManager.requestEnableNotification()

      // 用户操作后再检查一次结果
      return await notificationManager.isNotificationEnabled()
    } catch (e) {
      // 任意异常都视为不可用，避免影响后续逻辑
      hilog.error(DOMAIN, TAG, 'ensureEnabled err=%{public}s', JSON.stringify(e))
      return false
    }
  }

  // 发布一条行情通知：点击后可跳转回应用，并附带震动反馈
  static async publish(p: QuoteAlertPayload): Promise<void> {
    // 确保通知权限已开启（未开启会拉起系统授权页）
    const ok = await NotifyCenter.ensureEnabled()
    if (!ok) return

    try {
      // 构造 WantAgent：用于通知被点击时跳转回指定 Ability
      const wantAgentInfo: WantAgentConfig = {
        wants: [{
          bundleName: p.bundleName,     // 目标应用包名
          abilityName: p.abilityName,   // 打开的 Ability
        }],
        operationType: wantAgent.OperationType.START_ABILITY,
        requestCode: p.id,
        wantAgentFlags: [
          wantAgent.WantAgentFlags.UPDATE_PRESENT_FLAG
        ],
      }

      // 获取 WantAgent 实例
      const agent = await wantAgent.getWantAgent(wantAgentInfo)

      // 构造通知请求
      const req: notificationManager.NotificationRequest = {
        id: p.id,
        slotType: notification.SlotType.SOCIAL_COMMUNICATION,
        content: {
          notificationContentType:
          notificationManager.ContentType.NOTIFICATION_CONTENT_BASIC_TEXT,
          normal: {
            title: p.title,
            text: p.text,
            additionalText: p.additionalText,
          },
        },
        wantAgent: agent, // 绑定点击行为
      }

      // 发布系统通知
      await notificationManager.publish(req)
      hilog.info(DOMAIN, TAG, 'publish ok id=%{public}d', req.id)

      // 成功后给一次轻震动反馈
      await Haptics.vibrateShort()
    } catch (e) {
      // 捕获并记录通知发布失败的错误
      const errObj = e as ErrorObject
      const codeNum = Number(errObj.code)
      hilog.error(
        DOMAIN,
        TAG,
        'publish failed code=%{public}d err=%{public}s',
        codeNum,
        JSON.stringify(e),
      )
    }
  }

}
