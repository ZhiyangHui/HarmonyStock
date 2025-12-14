const storage = new LocalStorage()

interface RouterParams {
  formId: string
}

interface RouterActionOption {
  action: 'router'
  abilityName: string
  params: RouterParams
}

declare function postCardAction(ctx: Object, option: RouterActionOption): void

@Entry(storage)
@Component
struct QuoteWidgetSmall {
  @LocalStorageProp('formId') formId: string = ''
  @LocalStorageProp('title') title: string = ''
  @LocalStorageProp('code') code: string = ''
  @LocalStorageProp('name') name: string = ''
  @LocalStorageProp('priceText') priceText: string = '--'
  @LocalStorageProp('changeText') changeText: string = '--'
  @LocalStorageProp('pctText') pctText: string = '--'
  @LocalStorageProp('tsText') tsText: string = ''

  readonly abilityName: string = 'EntryAbility'

  private isUp(): boolean {
    if (this.changeText === '--') return true
    return !this.changeText.startsWith('-')
  }

  private openApp(): void {
    postCardAction(this, {
      action: 'router',
      abilityName: this.abilityName,
      params: {
        formId: this.formId
      }
    })
  }

  build() {
    Column() {
      Text(this.title)
        .fontSize(10)
        .fontColor('#BBBBBB')

      Text(this.name)
        .fontSize(14)
        .fontColor('#FFFFFF')
        .margin({ top: 6 })

      Text(this.code)
        .fontSize(10)
        .fontColor('#666666')

      Text(this.priceText)
        .fontSize(20)
        .fontWeight(FontWeight.Bold)
        .fontColor(this.isUp() ? '#FF4D4F' : '#39D98A')
        .margin({ top: 8 })

      Text(`${this.changeText}  ${this.pctText}`)
        .fontSize(11)
        .fontColor(this.isUp() ? '#FF7875' : '#6BE49B')
        .margin({ top: 4 })

      Text(this.tsText)
        .fontSize(10)
        .fontColor('#666666')
        .margin({ top: 8 })
    }
    .padding(12)
    .width('100%')
    .height('100%')
    .backgroundColor('#151515')
    .borderRadius(16)
    .onClick(() => {
      this.openApp()
    })
  }
}
